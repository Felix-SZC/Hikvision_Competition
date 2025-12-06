#!/usr/bin/env python
"""
可视化功能演示脚本
展示热力图和异常区域框选功能
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# 添加code目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from code.utils import visualize_anomaly_detection


def demo_visualization():
    """
    演示可视化功能
    """
    print("=== 异常检测可视化演示 ===\n")

    # 创建演示数据
    print("1. 创建演示图像和异常映射...")

    # 创建一个简单的测试图像（模拟工业产品）
    demo_image = Image.new('RGB', (224, 224), color=(200, 200, 200))

    # 绘制一些简单的形状来模拟产品
    from PIL import ImageDraw
    draw = ImageDraw.Draw(demo_image)
    # 绘制圆形产品主体
    draw.ellipse([50, 50, 174, 174], fill=(150, 150, 150), outline=(100, 100, 100))
    # 绘制一些特征
    draw.rectangle([80, 80, 144, 144], fill=(120, 120, 120))
    draw.ellipse([100, 100, 124, 124], fill=(180, 180, 180))

    # 创建模拟异常映射
    anomaly_map = np.zeros((224, 224), dtype=np.float32)

    # 添加一些异常区域（模拟缺陷）
    # 异常区域1：右上角的划痕
    anomaly_map[30:50, 180:200] = 0.8
    # 异常区域2：左下角的斑点
    anomaly_map[160:180, 40:60] = 0.6
    # 异常区域3：中间的小缺陷
    anomaly_map[100:120, 100:120] = 0.9

    # 添加一些噪声
    noise = np.random.rand(224, 224) * 0.2
    anomaly_map += noise
    anomaly_map = np.clip(anomaly_map, 0, 1)

    print("2. 生成可视化结果...")

    # 设置输出路径
    output_dir = "demo_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_anomaly_detection.png")

    # 执行可视化
    anomaly_score = 0.75  # 模拟异常分数
    threshold = 0.5       # 模拟阈值

    # 先计算异常区域数量
    from code.utils.visualize import find_anomaly_regions
    boxes = find_anomaly_regions(anomaly_map, threshold=threshold, min_area=10)

    result_image = visualize_anomaly_detection(
        original_image=demo_image,
        anomaly_map=anomaly_map,
        anomaly_score=anomaly_score,
        threshold=threshold,
        save_path=output_path,
        show=False,
        alpha=0.4,
        min_area=10
    )

    print(f"3. 可视化结果已保存到: {output_path}")
    print("\n可视化说明:")
    print("- 彩色热力图叠加在原图上（红色表示高异常概率）")
    print("- 红色框标出检测到的异常区域")
    print("- 左上角显示异常分数、阈值、状态和异常区域数量")
    print(f"- 当前演示检测到 {len(boxes)} 个异常区域")

    # 显示配置信息
    print("\n当前可视化配置:")
    from code.config import VISUALIZATION_CONFIG
    for key, value in VISUALIZATION_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n=== 演示完成 ===")


def test_real_data():
    """
    测试真实数据（如果有的话）
    """
    print("\n=== 测试真实数据 ===")

    # 检查是否有真实数据
    dataset_path = "dataset"
    model_path = "models"

    if not os.path.exists(dataset_path):
        print(f"未找到数据集目录: {dataset_path}")
        return

    if not os.path.exists(model_path):
        print(f"未找到模型目录: {model_path}")
        return

    print("发现数据集和模型，开始测试真实数据...")
    print("运行命令: cd code && python inference.py")
    print("注意：请确保已安装所有依赖包")


if __name__ == "__main__":
    demo_visualization()
    test_real_data()
