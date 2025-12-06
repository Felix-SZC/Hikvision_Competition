"""
可视化工具模块
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw
import os
import torch

# 动态导入配置，根据当前运行环境决定导入路径
try:
    from config import VISUALIZATION_CONFIG
except ImportError:
    # 如果从utils目录内运行，使用相对导入
    from ..config import VISUALIZATION_CONFIG


def anomaly_map_to_heatmap(anomaly_map, colormap='jet'):
    """
    将异常映射转换为热力图

    Args:
        anomaly_map: 异常映射 (H, W) 或 (1, H, W)
        colormap: 颜色映射名称

    Returns:
        heatmap: PIL Image对象
    """
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[0]  # 移除通道维度

    # 归一化到0-1范围
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    # 应用颜色映射
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(anomaly_map)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # 移除alpha通道并转换为uint8

    # 转换为PIL Image
    heatmap_pil = Image.fromarray(heatmap)

    return heatmap_pil


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4):
    """
    将热力图叠加到原图上

    Args:
        original_image: PIL Image对象
        heatmap: PIL Image对象
        alpha: 透明度

    Returns:
        overlayed_image: PIL Image对象
    """
    # 确保图像大小一致
    heatmap = heatmap.resize(original_image.size)

    # 将PIL图像转换为numpy数组
    original_array = np.array(original_image)
    heatmap_array = np.array(heatmap)

    # 叠加图像
    overlayed = cv2.addWeighted(original_array, 1 - alpha, heatmap_array, alpha, 0)

    # 转换回PIL Image
    overlayed_image = Image.fromarray(overlayed)

    return overlayed_image


def find_anomaly_regions(anomaly_map, threshold=None, min_area=10, red_threshold=None):
    """
    找到异常区域并返回边界框

    Args:
        anomaly_map: 异常映射 (H, W)
        threshold: 阈值，如果为None则自动计算
        min_area: 最小区域面积
        red_threshold: 红色区域阈值 (0-255)，控制框出敏感度

    Returns:
        boxes: 边界框列表 [(x1, y1, x2, y2), ...]
    """
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[0]

    # 使用配置文件中的默认值
    if red_threshold is None:
        red_threshold = VISUALIZATION_CONFIG['red_threshold']
    if min_area is None:
        min_area = VISUALIZATION_CONFIG['min_area']

    # 归一化异常映射到0-255范围
    anomaly_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    anomaly_255 = (anomaly_normalized * 255).astype(np.uint8)

    # 使用红色阈值来确定要框出的区域
    # 值越小，框出的区域越多（包括黄色区域）
    # 值越大，只框最红的区域
    binary = (anomaly_255 >= red_threshold).astype(np.uint8) * 255

    # 形态学操作：去除噪声并连接相邻区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 开运算：先腐蚀后膨胀，去除小噪声
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算：先膨胀后腐蚀，填充小空洞
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # 过滤掉过小的边框（可能是噪声）
            if w >= 5 and h >= 5:  # 最小宽度和高度
                boxes.append((x, y, x + w, y + h))

    return boxes


def draw_boxes_on_image(image, boxes, color='red', width=2):
    """
    在图像上绘制边界框

    Args:
        image: PIL Image对象
        boxes: 边界框列表 [(x1, y1, x2, y2), ...]
        color: 框的颜色
        width: 框的宽度

    Returns:
        image_with_boxes: 绘制了框的PIL Image对象
    """
    # 创建可绘制对象
    draw = ImageDraw.Draw(image)

    for box in boxes:
        x1, y1, x2, y2 = box
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    return image


def visualize_anomaly_detection(original_image, anomaly_map, anomaly_score, threshold=None,
                               save_path=None, show=False, alpha=0.4, min_area=10):
    """
    完整异常检测可视化流程

    Args:
        original_image: PIL Image对象
        anomaly_map: 异常映射 (H, W) 或 (1, H, W)
        anomaly_score: 异常分数
        threshold: 异常阈值
        save_path: 保存路径
        show: 是否显示图像
        alpha: 热力图透明度
        min_area: 最小异常区域面积
    """
    try:
        # 生成热力图
        heatmap = anomaly_map_to_heatmap(anomaly_map, colormap=VISUALIZATION_CONFIG['colormap'])

        # 叠加热力图到原图
        overlayed_image = overlay_heatmap_on_image(original_image, heatmap, alpha=alpha)

        # 找到异常区域
        boxes = find_anomaly_regions(anomaly_map, threshold=threshold, min_area=min_area, red_threshold=VISUALIZATION_CONFIG['red_threshold'])

        # 在叠加图像上绘制边界框
        final_image = draw_boxes_on_image(overlayed_image, boxes, color='red', width=3)

        # 添加分数信息
        draw = ImageDraw.Draw(final_image)

        # 设置文本样式
        font_size = 12  # 进一步减小字体大小
        try:
            # 尝试使用系统字体
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # 如果字体不可用，使用默认字体
            font = None

        # 准备文本
        score_text = f"Score: {anomaly_score:.4f}"
        threshold_text = f"Threshold: {threshold:.4f}" if threshold is not None else "Threshold: Auto"
        status_text = "Anomaly" if anomaly_score >= (threshold if threshold is not None else 0) else "Normal"
        boxes_text = f"Anomaly Regions: {len(boxes)}"

        # 设置文本颜色（异常用红色，正常用绿色）
        is_anomaly = anomaly_score >= (threshold if threshold is not None else 0)
        text_color = (255, 0, 0) if is_anomaly else (0, 255, 0)  # RGB颜色

        # 计算文本位置
        y_offset = 8
        text_positions = [
            (8, y_offset),
            (8, y_offset + 18),  # 进一步减小行间距
            (8, y_offset + 36),
            (8, y_offset + 54)
        ]

        # 直接绘制文本（无背景）
        for text, pos in zip([score_text, threshold_text, status_text, boxes_text], text_positions):
            if font:
                draw.text(pos, text, fill=text_color, font=font)
            else:
                draw.text(pos, text, fill=text_color)

        # 保存图像
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            final_image.save(save_path)
            print(f"可视化结果已保存: {save_path}")

        # 显示图像
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(final_image)
            plt.axis('off')
            plt.title(f"Anomaly Detection - Score: {anomaly_score:.4f}")
            plt.show()

        return final_image

    except Exception as e:
        print(f"可视化过程中出错: {e}")
        # 如果出错，至少保存原始图像
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                original_image.save(save_path)
                print(f"保存了原始图像: {save_path}")
            except Exception as save_error:
                print(f"保存图像时出错: {save_error}")
        return original_image
