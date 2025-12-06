import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from sklearn.random_projection import SparseRandomProjection
import tqdm
import math
import pickle

class PatchCore(nn.Module):
    """
    PatchCore "完全体" (Best & Classic Version)
    
    遵循 CVPR 2022 官方论文核心：
    1. Backbone: WideResNet50
    2. Layers: Layer 2 + Layer 3
    3. Pre-processing: LNA (Local Neighborhood Aggregation)
    4. Post-processing: Gaussian Blur (Sigma=4) -> 这是最关键的优化
    """
    
    def __init__(self, backbone_name='wide_resnet50_2', coreset_sampling_ratio=0.01, device='cuda'):
        super(PatchCore, self).__init__()
        self.device = device
        self.coreset_sampling_ratio = coreset_sampling_ratio
        
        # 1. 加载骨干网络
        print(f"正在加载骨干网络: {backbone_name}")
        if backbone_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_name == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"未知的骨干网络: {backbone_name}")
            
        self.model.to(device)
        self.model.eval()
        
        self.memory_bank = None
        self.threshold = None
        self.min_val = None
        self.max_val = None
        self.features = {}
        self._register_hooks()
        
        # 随机投影用于加速核心集选择
        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)
        
        # 2. 初始化高斯模糊核 (Classic PatchCore standard: kernel=33, sigma=4)
        self.gaussian_kernel = self._create_gaussian_kernel(kernel_size=33, sigma=4).to(device)

    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        self.model.layer2.register_forward_hook(hook_fn('layer2'))
        self.model.layer3.register_forward_hook(hook_fn('layer3'))

    def _create_gaussian_kernel(self, kernel_size=33, sigma=4):
        """
        创建高斯模糊核 (PyTorch版)，用于平滑异常热力图
        """
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # Reshape to (C_out, C_in, H, W) -> (1, 1, H, W) for convolution
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    def _embed(self, images):
        """特征提取 + LNA"""
        with torch.no_grad():
            _ = self.model(images)
        
        layer2 = self.features['layer2']
        layer3 = self.features['layer3']
        
        # LNA: 3x3 Average Pooling
        avg_pool = torch.nn.AvgPool2d(3, 1, 1)
        layer2 = avg_pool(layer2)
        layer3 = avg_pool(layer3)
        
        # Resize layer3 to layer2 size
        layer3 = F.interpolate(layer3, size=layer2.shape[2:], mode='bilinear', align_corners=False)
        
        # Concat
        embedding = torch.cat([layer2, layer3], dim=1)
        return embedding

    def fit(self, train_loader):
        """构建记忆库"""
        self.model.eval()
        embedding_list = []
        print("开始提取训练集特征...")
        
        with torch.no_grad():
            for images, _, _, _ in tqdm.tqdm(train_loader, desc="Fitting"):
                images = images.to(self.device)
                embedding = self._embed(images)
                # (B, C, H, W) -> (B*H*W, C)
                embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])
                embedding_list.append(embedding.cpu())
            
        full_embedding = torch.cat(embedding_list, dim=0)
        
        if self.coreset_sampling_ratio < 1.0:
            print(f"开始核心集采样 (Ratio: {self.coreset_sampling_ratio})...")
            self.memory_bank = self._approximate_coreset(full_embedding, self.coreset_sampling_ratio)
        else:
            self.memory_bank = full_embedding
            
        self.memory_bank = self.memory_bank.to(self.device)
        print(f"记忆库构建完成，形状: {self.memory_bank.shape}")

    def _approximate_coreset(self, embedding, ratio):
        """使用随机投影 + 贪心算法进行下采样"""
        # 投影到低维空间加速计算
        print("执行随机投影...")
        self.random_projector.fit(embedding.numpy())
        projected_embedding = torch.tensor(
            self.random_projector.transform(embedding.numpy()), 
            dtype=torch.float32
        )
        
        n_samples = embedding.shape[0]
        n_coreset = int(n_samples * ratio)
        
        # 初始化
        selected_indices = []
        idx = np.random.randint(n_samples)
        selected_indices.append(idx)
        
        # 初始距离
        current_points = projected_embedding[idx:idx+1] # (1, dim)
        # 计算所有点到当前选中点的距离
        dists = torch.cdist(projected_embedding, current_points).squeeze()
        
        print(f"开始贪心选择 {n_coreset} 个样本...")
        for _ in tqdm.tqdm(range(n_coreset - 1)):
            # 选距离最大的点
            idx = torch.argmax(dists).item()
            selected_indices.append(idx)
            
            # 更新距离：只更新新选中的点带来的更短距离
            current_dist = torch.cdist(projected_embedding, projected_embedding[idx:idx+1]).squeeze()
            dists = torch.min(dists, current_dist)
            
        return embedding[selected_indices]

    def predict(self, images):
        """
        预测逻辑 - 加入了 Classic 的高斯平滑
        """
        self.model.eval()
        images = images.to(self.device)
        
        with torch.no_grad():
            embedding = self._embed(images) # (B, C, H, W)
            B, C, H, W = embedding.shape
            
            # 1. 计算距离 (Nearest Neighbor Search)
            query = embedding.permute(0, 2, 3, 1).reshape(-1, C) # (N, C)
            
            # 分块计算防止OOM
            dists_list = []
            chunk_size = 2048 
            for i in range(0, query.shape[0], chunk_size):
                q_chunk = query[i:i+chunk_size]
                # 计算到记忆库的距离
                d = torch.cdist(q_chunk, self.memory_bank) 
                # 取最近邻
                min_d, _ = torch.min(d, dim=1)
                dists_list.append(min_d)
            
            patch_scores = torch.cat(dists_list) # (B*H*W)
            
            # 2. 重塑回图片形状
            anomaly_map = patch_scores.reshape(B, 1, H, W)
            
            # 3. 双线性插值还原到原图尺寸 (224x224)
            anomaly_map = F.interpolate(
                anomaly_map, 
                size=(images.shape[2], images.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 4. [核心改进] 高斯平滑 (Gaussian Blur)
            # 使用 padding 保持尺寸不变
            padding = self.gaussian_kernel.shape[-1] // 2
            anomaly_map = F.conv2d(
                anomaly_map, 
                self.gaussian_kernel, 
                padding=padding
            )
            
            # 5. 图像级分数：取热力图的最大值
            # 经过平滑后的最大值比原始最大值更鲁棒
            anomaly_score = torch.amax(anomaly_map, dim=(1, 2, 3))
            
            return anomaly_score, anomaly_map

    def save_model(self, path):
        """保存模型到文件"""
        print(f"正在保存模型到 {path}...")
        state = {
            'memory_bank': self.memory_bank.cpu(),
            'threshold': self.threshold,
            'min_val': self.min_val,
            'max_val': self.max_val
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print("模型保存完成")
            
    def load_model(self, path):
        """从文件加载模型"""
        print(f"正在从 {path} 加载模型...")
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.memory_bank = state['memory_bank'].to(self.device)
        self.threshold = state.get('threshold', 0.5)
        self.min_val = state.get('min_val')
        self.max_val = state.get('max_val')
        print(f"模型加载完成 - 记忆库形状: {self.memory_bank.shape}, 阈值: {self.threshold}")
