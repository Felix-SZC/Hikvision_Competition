"""
配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
DATASET_ROOT = PROJECT_ROOT / 'dataset'
MODEL_DIR = PROJECT_ROOT / 'models'
VISUALIZATION_DIR = PROJECT_ROOT / 'visualizations'

# 模型配置
MODEL_CONFIG = {
    'backbone_name': 'wide_resnet50_2',  # 'resnet18' 或 'wide_resnet50_2'
    'coreset_sampling_ratio': 0.1,  # Coreset采样比例
    'device': 'cuda',  # 'cuda' 或 'cpu'
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 16,
    'num_workers': 0,  # Windows上建议设为0
    'train_split_ratio': 0.8,  # 训练集比例（剩余用于验证）
    'defect_samples_for_threshold': 10,  # 用于阈值搜索的缺陷样本数量
}

# 数据预处理配置
DATA_CONFIG = {
    'image_size': (224, 224),
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    # 数据增强配置 - 提高对光照变化的鲁棒性
    'use_augmentation': True,
    'brightness_range': [0.7, 1.3],  # 亮度增强范围，降低下限提高对暗光适应
    'contrast_range': [0.8, 1.2],    # 对比度增强范围
    'saturation_range': [0.8, 1.2],  # 饱和度增强范围
    'hue_range': [-0.1, 0.1],        # 色调增强范围
    'random_crop_prob': 0.5,         # 随机裁剪概率
    'gaussian_noise_prob': 0.3,      # 高斯噪声概率
    'gaussian_noise_std': 0.05,      # 高斯噪声标准差
}

# 评估配置
EVAL_CONFIG = {
    'batch_size': 1,
    'visualize': True,  # 是否在评估时生成可视化
    'max_vis_samples': 20,  # 每个类别最多可视化的样本数，设置为-1表示生成所有样本
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'colormap': 'jet',
    'alpha': 0.4,  # 热力图透明度
    'min_area': 10,  # 最小异常区域面积
    'red_threshold': 200,  # 红色区域阈值（0-255），值越小框越多，值越大只框最红的
                           # 建议范围：150-220，可根据实际情况调整
                           # 如果很多红色区域没框出来，降低这个值（如150-170）
                           # 如果框了太多黄色区域，提高这个值（如200-220）
}

# 类别列表
CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 创建必要的目录
for dir_path in [MODEL_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

