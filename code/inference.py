"""
推理/评估脚本
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import time

# 处理导入路径：支持从项目根目录或code目录运行
if __name__ == '__main__':
    # 如果从code目录运行，添加父目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, parent_dir)

try:
    from code.data import MVTecDataset
    from code.models import PatchCore
    from code.utils import compute_metrics, visualize_anomaly_detection
    from code.config import (
        DATASET_ROOT, MODEL_DIR, VISUALIZATION_DIR,
        MODEL_CONFIG, DATA_CONFIG, EVAL_CONFIG, TRAIN_CONFIG, CATEGORIES
    )
except ImportError:
    # 如果从code目录内运行，使用相对导入
    from data import MVTecDataset
    from models import PatchCore
    from utils import compute_metrics, visualize_anomaly_detection
    from config import (
        DATASET_ROOT, MODEL_DIR, VISUALIZATION_DIR,
        MODEL_CONFIG, DATA_CONFIG, EVAL_CONFIG, TRAIN_CONFIG, CATEGORIES
    )

def evaluate(dataset_path, category_ids, model_dir, device='cuda', visualize=False, vis_output_dir=None):
    if vis_output_dir is None:
        vis_output_dir = str(VISUALIZATION_DIR)
    
    transform = transforms.Compose([
        transforms.Resize(DATA_CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATA_CONFIG['mean'], std=DATA_CONFIG['std'])
    ])

    total_f1 = 0
    count = 0

    for cat_id in category_ids:
        print(f"\n--- 正在评估类别: {cat_id} ---")
        
        model_path = os.path.join(model_dir, f"model_{cat_id}.pt")
        if not os.path.exists(model_path):
            print(f"未找到类别 {cat_id} 的模型。跳过。")
            continue
            
        # Load Model
        print(f"正在加载模型: {model_path}")
        try:
            model = PatchCore(backbone_name='wide_resnet50_2', device=device)
            model.load_model(model_path)
            print("模型加载成功")
        except Exception as e:
            print(f"加载类别 {cat_id} 的模型时出错: {e}")
            continue
        
        # Load Test Data
        print(f"正在加载测试数据...")
        test_dataset = MVTecDataset(
            dataset_path, cat_id, split='test', 
            transform=transform, 
            defect_samples_for_threshold=TRAIN_CONFIG['defect_samples_for_threshold']
        )
        
        if len(test_dataset) == 0:
            print(f"类别 {cat_id} 没有测试数据。")
            continue
        
        # 统计测试数据中的正常和异常样本
        normal_test = sum(1 for _, label, _, _ in test_dataset if label == 0)
        abnormal_test = sum(1 for _, label, _, _ in test_dataset if label == 1)
        print(f"测试数据统计 - 正常样本: {normal_test}, NG样本: {abnormal_test}, 总计: {len(test_dataset)}")
            
        test_loader = DataLoader(
            test_dataset, 
            batch_size=EVAL_CONFIG['batch_size'], 
            shuffle=False
        )
        
        labels = []
        scores = []
        times = []
        
        print(f"正在评估 {len(test_dataset)} 个样本...")
        
        # 如果需要可视化
        if visualize:
            if not os.path.exists(vis_output_dir):
                os.makedirs(vis_output_dir)
            # 为每个类别创建单独的子文件夹
            category_vis_dir = os.path.join(vis_output_dir, f"category_{cat_id}")
            if not os.path.exists(category_vis_dir):
                os.makedirs(category_vis_dir)
            vis_count = 0
            max_vis_samples = EVAL_CONFIG['max_vis_samples']
        
        for idx, (images, label, _, img_path) in enumerate(test_loader):
            start_time = time.time()
            s, anomaly_map = model.predict(images)
            end_time = time.time()
            times.append(end_time - start_time)
            
            scores.extend(s.detach().cpu().numpy())
            labels.extend(label.numpy())
            
            # 可视化（仅对前几个样本，如果max_vis_samples=-1则生成所有样本）
            if visualize and (max_vis_samples == -1 or vis_count < max_vis_samples):
                original_image = Image.open(img_path[0]).convert('RGB')
                original_image = original_image.resize((224, 224))
                anomaly_score = s.item()
                anomaly_map_single = anomaly_map[0, 0]  # (H, W)
                
                vis_path = os.path.join(category_vis_dir, 
                                       f"sample_{idx}_score_{anomaly_score:.4f}.png")
                visualize_anomaly_detection(
                    original_image,
                    anomaly_map_single,
                    anomaly_score,
                    threshold=model.threshold,
                    save_path=vis_path,
                    show=False,
                    alpha=0.4,
                    min_area=50  # 最小区域面积，过滤小噪声，减少误检
                )
                vis_count += 1
                if vis_count == 1:
                    print(f"  可视化结果已保存到: {category_vis_dir}")
            
        if len(times) > 0:
            avg_time = sum(times) / len(times)
            print(f"平均推理时间: {avg_time:.4f}秒")
        
        # Compute Metrics
        labels_array = np.array(labels)
        scores_array = np.array(scores)
        preds = (scores_array >= model.threshold).astype(int)
        
        # 计算详细指标
        tp = np.sum((preds == 1) & (labels_array == 1))
        fp = np.sum((preds == 1) & (labels_array == 0))
        tn = np.sum((preds == 0) & (labels_array == 0))
        fn = np.sum((preds == 0) & (labels_array == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
        
        f1 = compute_metrics(labels, scores, threshold=model.threshold)
        
        print(f"\n评估结果 (阈值: {model.threshold:.4f}):")
        print(f"  真阳性(TP): {tp}, 假阳性(FP): {fp}, 真阴性(TN): {tn}, 假阴性(FN): {fn}")
        print(f"  精确率(Precision): {precision:.4f}")
        print(f"  召回率(Recall): {recall:.4f}")
        print(f"  准确率(Accuracy): {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")
        
        total_f1 += f1
        count += 1
        
    if count > 0:
        print(f"\n所有评估类别的平均F1分数: {total_f1/count:.4f}")

if __name__ == '__main__':
    dataset_root = os.path.join(os.getcwd(), 'dataset') 
    output_models = os.path.join(os.getcwd(), 'models')
    
    if not os.path.exists(dataset_root) and os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'dataset')):
        dataset_root = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
        output_models = os.path.join(os.path.dirname(os.getcwd()), 'models')

    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"数据集根目录: {dataset_root}")
    print(f"模型目录: {output_models}")
    print(f"开始评估，共 {len(categories)} 个类别")
    print("=" * 60)
    
    start_time = time.time()
    evaluate(
        dataset_root, categories, output_models, 
        device=device,
        visualize=EVAL_CONFIG['visualize']
    )
    end_time = time.time()
    
    print("=" * 60)
    print(f"评估完成！总耗时: {end_time - start_time:.2f} 秒 ({(end_time - start_time)/60:.2f} 分钟)")

