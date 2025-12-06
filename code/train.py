"""
训练脚本
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import random
import time

# 数据增强类
class IndustrialAugmentation:
    """工业场景数据增强，提高对光照和噪声的鲁棒性"""

    def __init__(self, config):
        self.config = config

    def __call__(self, img):
        # 随机亮度调整
        if random.random() < 0.8:  # 80%概率应用亮度增强
            brightness_factor = random.uniform(
                self.config['brightness_range'][0],
                self.config['brightness_range'][1]
            )
            img = transforms.functional.adjust_brightness(img, brightness_factor)

        # 随机对比度调整
        if random.random() < 0.6:  # 60%概率应用对比度增强
            contrast_factor = random.uniform(
                self.config['contrast_range'][0],
                self.config['contrast_range'][1]
            )
            img = transforms.functional.adjust_contrast(img, contrast_factor)

        # 随机饱和度调整
        if random.random() < 0.4:  # 40%概率应用饱和度增强
            saturation_factor = random.uniform(
                self.config['saturation_range'][0],
                self.config['saturation_range'][1]
            )
            img = transforms.functional.adjust_saturation(img, saturation_factor)

        # 随机色调调整
        if random.random() < 0.3:  # 30%概率应用色调增强
            hue_factor = random.uniform(
                self.config['hue_range'][0],
                self.config['hue_range'][1]
            )
            img = transforms.functional.adjust_hue(img, hue_factor)

        # 随机高斯噪声
        if random.random() < self.config.get('gaussian_noise_prob', 0.3):
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.config.get('gaussian_noise_std', 0.05),
                                   img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            img = transforms.functional.to_pil_image((img_array * 255).astype(np.uint8))

        return img


def get_train_transform(config, augment=True):
    """获取训练数据变换，支持数据增强"""
    base_transforms = [
        transforms.Resize(config['image_size']),
    ]

    if augment and config.get('use_augmentation', False):
        # 添加数据增强来提高对光照变化的鲁棒性
        base_transforms.append(IndustrialAugmentation(config))

    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    return transforms.Compose(base_transforms)


# 处理导入路径：支持从项目根目录或code目录运行
if __name__ == '__main__':
    # 如果从code目录运行，添加父目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# 统一导入，利用已修改的sys.path
from data.dataset import MVTecDataset
from models.patchcore import PatchCore
from utils import search_best_threshold
from config import (
    DATASET_ROOT, MODEL_DIR, MODEL_CONFIG, TRAIN_CONFIG,
    DATA_CONFIG, CATEGORIES
)


def train(dataset_path, category_ids, output_dir, device='cuda'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Transform
    transform = transforms.Compose([
        transforms.Resize(DATA_CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATA_CONFIG['mean'], std=DATA_CONFIG['std'])
    ])

    for cat_id in category_ids:
        print(f"\n--- 正在处理类别: {cat_id} ---")
        
        # 1. Prepare Data
        print(f"正在加载类别 {cat_id} 的训练数据...")
        try:
            full_train_dataset = MVTecDataset(dataset_path, cat_id, split='train', transform=transform)
        except Exception as e:
            print(f"由于错误跳过类别 {cat_id}: {e}")
            continue
            
        if len(full_train_dataset) == 0:
            print(f"未找到类别 {cat_id} 的训练数据。跳过。")
            continue
        
        print(f"成功加载 {len(full_train_dataset)} 个正常训练样本")

        # Split normal data for Memory Bank and Validation
        num_train = len(full_train_dataset)
        indices = list(range(num_train))
        random.shuffle(indices)
        
        # 使用配置的train_split_ratio
        split_idx = int(num_train * TRAIN_CONFIG['train_split_ratio'])
        if split_idx == num_train: # If very few samples
            split_idx = max(1, num_train - 5)  # 至少保留5个验证样本
            
        bank_indices = indices[:split_idx]
        val_normal_indices = indices[split_idx:]
        
        print(f"训练样本数: {len(bank_indices)}, 验证正常样本数: {len(val_normal_indices)}")
        
        bank_dataset = Subset(full_train_dataset, bank_indices)
        val_normal_dataset = Subset(full_train_dataset, val_normal_indices)
        
        bank_loader = DataLoader(
            bank_dataset, 
            batch_size=TRAIN_CONFIG['batch_size'], 
            shuffle=False, 
            num_workers=TRAIN_CONFIG['num_workers']
        )
        val_normal_loader = DataLoader(
            val_normal_dataset, 
            batch_size=TRAIN_CONFIG['batch_size'], 
            shuffle=False
        )
        
        # Load Threshold NG samples
        print(f"正在加载用于阈值搜索的NG样本...")
        threshold_ng_dataset = MVTecDataset(
            dataset_path, cat_id, split='threshold', 
            transform=transform, 
            defect_samples_for_threshold=TRAIN_CONFIG['defect_samples_for_threshold']
        )
        if len(threshold_ng_dataset) == 0:
            print("警告: 未找到用于阈值搜索的NG样本。")
        else:
            print(f"成功加载 {len(threshold_ng_dataset)} 个NG样本用于阈值搜索")
            
        threshold_ng_loader = DataLoader(threshold_ng_dataset, batch_size=1, shuffle=False)
        
        # 2. Initialize Model
        model = PatchCore(
            backbone_name=MODEL_CONFIG['backbone_name'],
            coreset_sampling_ratio=MODEL_CONFIG['coreset_sampling_ratio'],
            device=device
        )
        
        # 3. Fit (Build Memory Bank)
        model.fit(bank_loader)
        
        # 4. Threshold Search
        print("正在计算阈值搜索的分数...")
        labels = []
        scores = []
        
        # Normal samples
        normal_count = 0
        if len(val_normal_dataset) > 0:
            print(f"正在处理 {len(val_normal_dataset)} 个验证正常样本...")
            for images, label, _, _ in val_normal_loader:
                s, _ = model.predict(images)
                scores.extend(s.detach().cpu().numpy())
                labels.extend(label.numpy())
                normal_count += len(label)
            print(f"已处理 {normal_count} 个正常样本")
            
        # Abnormal samples
        abnormal_count = 0
        if len(threshold_ng_dataset) > 0:
            print(f"正在处理 {len(threshold_ng_dataset)} 个NG样本...")
            for images, label, _, _ in threshold_ng_loader:
                s, _ = model.predict(images)
                scores.extend(s.detach().cpu().numpy())
                labels.extend(label.numpy())
                abnormal_count += len(label)
            print(f"已处理 {abnormal_count} 个NG样本")
        
        print(f"阈值搜索数据统计 - 正常样本: {normal_count}, NG样本: {abnormal_count}, 总计: {len(labels)}") 
            
        # Search
        if len(labels) > 0:
            # 打印分数分布信息，便于调试
            normal_scores = np.array(scores)[np.array(labels) == 0]
            abnormal_scores = np.array(scores)[np.array(labels) == 1]
            if len(normal_scores) > 0 and len(abnormal_scores) > 0:
                print(f"分数统计 - 正常样本: 均值={normal_scores.mean():.4f}, 最大值={normal_scores.max():.4f}, 95%分位={np.percentile(normal_scores, 95):.4f}")
                print(f"分数统计 - NG样本: 均值={abnormal_scores.mean():.4f}, 最小值={abnormal_scores.min():.4f}")
            
            # 使用改进的阈值搜索
            best_threshold, best_f1 = search_best_threshold(labels, scores, method='f1')
            model.threshold = best_threshold
            print(f"类别 {cat_id} - 最佳阈值: {best_threshold}, F1分数: {best_f1}")
        else:
            print("阈值搜索数据不足。使用默认值。")
            model.threshold = 10.0 # Safe high default or empirical value
        
        # 5. Save Model
        save_path = os.path.join(output_dir, f"model_{cat_id}.pt")
        model.save_model(save_path)
        print(f"模型已保存到 {save_path}")

if __name__ == '__main__':
    dataset_root = str(DATASET_ROOT)
    output_models = str(MODEL_DIR)
    categories = CATEGORIES
    
    device = MODEL_CONFIG['device'] if torch.cuda.is_available() and MODEL_CONFIG['device'] == 'cuda' else 'cpu'
    print(f"使用设备: {device}")
    print(f"数据集根目录: {dataset_root}")
    print(f"模型输出目录: {output_models}")
    print(f"开始训练，共 {len(categories)} 个类别")
    print("=" * 60)
    
    start_time = time.time()
    train(dataset_root, categories, output_models, device=device)
    end_time = time.time()
    
    print("=" * 60)
    print(f"训练完成！总耗时: {end_time - start_time:.2f} 秒 ({(end_time - start_time)/60:.2f} 分钟)")

