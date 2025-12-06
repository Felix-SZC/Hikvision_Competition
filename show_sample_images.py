#!/usr/bin/env python
"""
显示数据集样本图片信息
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def show_sample_info():
    """显示一些样本图片的基本信息"""

    dataset_path = 'dataset'
    categories = ['1', '2', '3']

    print("=== 样本图片详细信息 ===\n")

    for category in categories:
        print(f"类别 {category} 样本:")

        # 正常样本
        ok_dir = os.path.join(dataset_path, category, f"{category}-ok")
        if os.path.exists(ok_dir):
            ok_files = [f for f in os.listdir(ok_dir) if f.endswith('.bmp')][:3]  # 只显示前3个
            print("  正常样本:")
            for i, filename in enumerate(ok_files, 1):
                filepath = os.path.join(ok_dir, filename)
                try:
                    img = Image.open(filepath)
                    print(f"    {i}. {filename}: {img.size}, 模式:{img.mode}")
                except Exception as e:
                    print(f"    {i}. {filename}: 读取失败 - {e}")

        # 异常样本
        ng_dir = os.path.join(dataset_path, category, f"{category}_ng")
        if os.path.exists(ng_dir):
            ng_files = [f for f in os.listdir(ng_dir) if f.endswith('.bmp') and not f.endswith('_t.bmp')][:3]
            print("  异常样本:")
            for i, filename in enumerate(ng_files, 1):
                filepath = os.path.join(ng_dir, filename)
                mask_file = filename.replace('.bmp', '_t.bmp')
                mask_path = os.path.join(ng_dir, mask_file)

                try:
                    img = Image.open(filepath)
                    has_mask = os.path.exists(mask_path)
                    print(f"    {i}. {filename}: {img.size}, 模式:{img.mode}, 有掩码:{has_mask}")
                except Exception as e:
                    print(f"    {i}. {filename}: 读取失败 - {e}")

        print()

def analyze_image_statistics():
    """分析图片的统计信息"""

    dataset_path = 'dataset'
    categories = ['1', '2', '3', '4', '5']

    print("=== 图片尺寸统计 ===\n")

    all_sizes = {'normal': [], 'anomaly': []}

    for category in categories:
        # 正常样本
        ok_dir = os.path.join(dataset_path, category, f"{category}-ok")
        if os.path.exists(ok_dir):
            ok_files = [f for f in os.listdir(ok_dir) if f.endswith('.bmp')][:10]  # 采样10张
            for filename in ok_files:
                filepath = os.path.join(ok_dir, filename)
                try:
                    img = Image.open(filepath)
                    all_sizes['normal'].append(img.size)
                except:
                    pass

        # 异常样本
        ng_dir = os.path.join(dataset_path, category, f"{category}_ng")
        if os.path.exists(ng_dir):
            ng_files = [f for f in os.listdir(ng_dir) if f.endswith('.bmp') and not f.endswith('_t.bmp')][:10]
            for filename in ng_files:
                filepath = os.path.join(ng_dir, filename)
                try:
                    img = Image.open(filepath)
                    all_sizes['anomaly'].append(img.size)
                except:
                    pass

    # 计算统计信息
    def get_stats(sizes):
        if not sizes:
            return None
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        return {
            'count': len(sizes),
            'avg_width': int(np.mean(widths)),
            'avg_height': int(np.mean(heights)),
            'min_size': min(sizes),
            'max_size': max(sizes)
        }

    normal_stats = get_stats(all_sizes['normal'])
    anomaly_stats = get_stats(all_sizes['anomaly'])

    print("正常样本统计:")
    if normal_stats:
        print(f"  样本数: {normal_stats['count']}")
        print(f"  平均尺寸: {normal_stats['avg_width']}x{normal_stats['avg_height']}")
        print(f"  尺寸范围: {normal_stats['min_size']} - {normal_stats['max_size']}")

    print("\n异常样本统计:")
    if anomaly_stats:
        print(f"  样本数: {anomaly_stats['count']}")
        print(f"  平均尺寸: {anomaly_stats['avg_width']}x{anomaly_stats['avg_height']}")
        print(f"  尺寸范围: {anomaly_stats['min_size']} - {anomaly_stats['max_size']}")

if __name__ == "__main__":
    show_sample_info()
    analyze_image_statistics()
