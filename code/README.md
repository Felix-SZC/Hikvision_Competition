# PatchCore 缺陷检测系统

基于改进版 PatchCore 算法，针对工业小样本缺陷检测赛道设计。

## 环境依赖

请确保安装以下 Python 库：
```bash
pip install -r requirements.txt
```
注意：代码中默认使用 CUDA，如果没有 GPU，会自动切换到 CPU（速度会慢）。

## 数据集准备

请确保数据集结构如下（与项目根目录平级或在项目根目录下）：
```
dataset/
  1/
    1-ok/
    1_ng/
      mark.txt
      ...
  ...
```

## 运行训练

训练脚本会遍历指定的数据集类别，构建特征库并搜索最佳阈值，模型保存在 `models/` 目录下。

```bash
python code/train.py
```

可以在 `code/train.py` 中修改 `categories` 列表来指定要训练的类别。

## 运行推理/评估

推理脚本会加载保存的模型，对测试集（NG样本）进行评估，输出 F1 Score 和 推理时间。

```bash
python code/inference.py
```

### 可视化功能 ✨

系统支持生成异常检测的可视化结果，包括热力图和异常区域框选。

#### 生成可视化结果

```bash
# 修改 code/config.py 中的配置
EVAL_CONFIG = {
    'visualize': True,  # 启用可视化
    'max_vis_samples': 20,  # 每个类别生成的可视化样本数，-1表示生成所有样本
}
```

然后运行推理：
```bash
python code/inference.py
```

可视化结果将保存在 `visualizations/` 目录中，每个图像包含：
- 原图叠加彩色热力图（红色表示高异常概率）
- 红色框标出检测到的异常区域
- 左上角显示异常分数、阈值、状态和异常区域数量

#### 可视化配置

在 `code/config.py` 中可以调整可视化参数：

```python
VISUALIZATION_CONFIG = {
    'colormap': 'jet',        # 热力图颜色方案
    'alpha': 0.4,             # 热力图透明度
    'min_area': 10,           # 最小异常区域面积
    'red_threshold': 200,     # 红色区域阈值(0-255)，控制框出敏感度
}
```

#### 演示脚本

运行演示脚本来查看可视化效果：

```bash
python demo_visualization.py
```

## 代码结构说明

- `dataset.py`: 数据加载，支持读取 `mark.txt` 标注。
- `patchcore.py`: PatchCore 算法实现，包含 WideResNet50 特征提取和 Coreset 采样。
- `train.py`: 训练流程，利用 90% 正常样本构建库，利用 5 个缺陷样本优化阈值。
- `inference.py`: 推理流程，加载模型进行预测。
- `utils.py`: 评估指标计算。

