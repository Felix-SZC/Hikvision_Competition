"""
评估指标模块
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve


def search_best_threshold(labels, scores, method='f1', percentile=95):
    """
    Search for the threshold that maximizes F1 score or uses statistical method.
    
    Args:
        labels: 标签数组
        scores: 分数数组
        method: 'f1' 或 'statistical'
        percentile: 如果使用statistical方法，使用正常样本的百分位数
    """
    labels = np.array(labels)
    scores = np.array(scores)
    
    if len(np.unique(labels)) < 2:
        print("警告: 标签中只有一个类别。使用统计方法。")
        method = 'statistical'
    
    if method == 'statistical':
        # 使用正常样本的统计分布
        normal_scores = scores[labels == 0]
        if len(normal_scores) > 0:
            # 使用正常样本的百分位数作为阈值
            threshold = np.percentile(normal_scores, percentile)
            # 计算在该阈值下的F1
            preds = (scores >= threshold).astype(int)
            f1 = f1_score(labels, preds)
            print(f"统计阈值 (正常样本{percentile}百分位): {threshold:.4f}, F1分数: {f1:.4f}")
            return threshold, f1
        else:
            # 如果没有正常样本，使用中位数
            threshold = np.median(scores)
            print(f"警告: 无正常样本，使用中位数作为阈值: {threshold:.4f}")
            return threshold, 0.0
    
    # 原始F1最大化方法
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * recall * precision / (recall + precision)
    
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    # thresholds is shorter than precision/recall by 1
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1]
        
    best_f1 = f1_scores[best_idx]
    
    # 检查过拟合：如果F1太高且样本量小，使用更稳健的方法
    normal_scores = scores[labels == 0]
    abnormal_scores = scores[labels == 1]
    
    if len(normal_scores) > 0 and len(abnormal_scores) > 0:
        # 计算正常样本和异常样本的分数分布
        normal_max = normal_scores.max()
        abnormal_min = abnormal_scores.min()
        
        # 如果F1太高（可能是过拟合），或者正常样本最高分 > 异常样本最低分（说明有重叠）
        if (best_f1 >= 0.99 and len(labels) < 30) or normal_max > abnormal_min:
            print(f"检测到可能过拟合或分数重叠，使用稳健的统计方法...")
            # 使用正常样本的百分位数，但要确保低于异常样本的最小值
            # 尝试多个百分位数，选择F1最好的
            candidate_percentiles = [90, 92, 95, 97, 99]
            best_stat_threshold = None
            best_stat_f1 = 0
            
            for p in candidate_percentiles:
                stat_threshold = np.percentile(normal_scores, p)
                # 确保阈值低于异常样本的最小值（留一些余量）
                if stat_threshold < abnormal_min * 0.9:  # 留10%余量
                    preds = (scores >= stat_threshold).astype(int)
                    stat_f1 = f1_score(labels, preds)
                    if stat_f1 > best_stat_f1:
                        best_stat_f1 = stat_f1
                        best_stat_threshold = stat_threshold
            
            # 如果统计方法找到更好的阈值，使用它
            if best_stat_threshold is not None and best_stat_f1 >= best_f1 * 0.9:  # 至少是F1方法的90%
                print(f"使用统计阈值: {best_stat_threshold:.4f}, F1: {best_stat_f1:.4f} (原F1方法: {best_f1:.4f})")
                return best_stat_threshold, best_stat_f1
    
    print(f"最佳阈值: {best_threshold:.4f}, 最大F1分数: {best_f1:.4f}")
    
    return best_threshold, best_f1


def compute_metrics(labels, scores, threshold=0.5):
    """
    计算评估指标
    
    Args:
        labels: 真实标签
        scores: 预测分数
        threshold: 分类阈值
    
    Returns:
        f1: F1分数
    """
    labels = np.array(labels)
    scores = np.array(scores)
    preds = (scores >= threshold).astype(int)
    
    f1 = f1_score(labels, preds)
    return f1

