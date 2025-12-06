#!/usr/bin/env python
"""
数据集分析脚本
分析Hikvision竞赛数据集的基本信息
"""

import os
from PIL import Image
import numpy as np

def analyze_dataset():
    """分析数据集结构和基本信息"""

    dataset_path = 'dataset'

    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return

    # 获取所有类别
    categories = [d for d in os.listdir(dataset_path)
                 if os.path.isdir(os.path.join(dataset_path, d))]
    categories.sort(key=int)

    print("=== Hikvision数据集分析 ===\n")
    print(f"总类别数: {len(categories)}")
    print(f"类别列表: {categories}\n")

    # 分析每个类别
    total_normal = 0
    total_anomaly = 0

    for category in categories:
        print(f"类别 {category}:")

        # 正常样本目录
        ok_dir = os.path.join(dataset_path, category, f"{category}-ok")
        # 异常样本目录
        ng_dir = os.path.join(dataset_path, category, f"{category}_ng")

        # 统计正常样本
        normal_count = 0
        if os.path.exists(ok_dir):
            normal_files = [f for f in os.listdir(ok_dir) if f.endswith('.bmp')]
            normal_count = len(normal_files)
            total_normal += normal_count

        # 统计异常样本
        anomaly_count = 0
        mask_count = 0
        if os.path.exists(ng_dir):
            all_files = os.listdir(ng_dir)
            bmp_files = [f for f in all_files if f.endswith('.bmp')]
            # 异常样本是不带_t后缀的bmp文件
            anomaly_files = [f for f in bmp_files if not f.endswith('_t.bmp')]
            # 掩码是带_t后缀的bmp文件
            mask_files = [f for f in bmp_files if f.endswith('_t.bmp')]

            anomaly_count = len(anomaly_files)
            mask_count = len(mask_files)
            total_anomaly += anomaly_count

        print(f"  正常样本: {normal_count} 张")
        print(f"  异常样本: {anomaly_count} 张")
        print(f"  异常掩码: {mask_count} 张")

        # 显示图片信息（取第一个样本）
        if normal_count > 0:
            sample_path = os.path.join(ok_dir, normal_files[0])
            try:
                img = Image.open(sample_path)
                print(f"  正常样本尺寸: {img.size}, 模式: {img.mode}")
            except Exception as e:
                print(f"  读取正常样本失败: {e}")

        if anomaly_count > 0:
            sample_path = os.path.join(ng_dir, anomaly_files[0])
            try:
                img = Image.open(sample_path)
                print(f"  异常样本尺寸: {img.size}, 模式: {img.mode}")
            except Exception as e:
                print(f"  读取异常样本失败: {e}")

        print()

    print("=== 数据集汇总 ===")
    print(f"总正常样本: {total_normal} 张")
    print(f"总异常样本: {total_anomaly} 张")
    print(f"总样本数: {total_normal + total_anomaly} 张")

    # 数据集特点分析
    print("\n=== 数据集特点 ===")
    print("1. 工业产品质量检测数据集")
    print("2. 包含10个不同类别的工业产品")
    print("3. 每个类别都有正常样本(合格品)和异常样本(不合格品)")
    print("4. 异常样本配有对应的缺陷掩码图像(_t.bmp后缀)")
    print("5. 图片格式为BMP，适合工业检测场景")
    print("6. 基于PatchCore算法进行异常检测")

if __name__ == "__main__":
    analyze_dataset()
