# 代码结构说明

## 目录结构

```
code/
├── __init__.py              # 包初始化文件
├── config.py                # 配置文件（统一管理所有配置）
├── train.py                 # 训练脚本
├── inference.py             # 推理/评估脚本
├── visualize_example.py     # 可视化示例脚本
│
├── models/                  # 模型模块
│   ├── __init__.py
│   └── patchcore.py        # PatchCore模型实现
│
├── data/                    # 数据模块
│   ├── __init__.py
│   └── dataset.py          # 数据集加载器
│
└── utils/                   # 工具模块
    ├── __init__.py
    ├── metrics.py          # 评估指标（阈值搜索、F1计算等）
    └── visualize.py        # 可视化功能（热力图、边界框等）
```

## 模块说明

### 1. config.py
统一管理所有配置参数，包括：
- 路径配置（数据集、模型、可视化输出）
- 模型配置（backbone、采样比例、设备）
- 训练配置（batch size、数据划分比例等）
- 评估配置
- 可视化配置

### 2. models/patchcore.py
PatchCore模型的核心实现：
- `PatchCore` 类：异常检测模型
- 特征提取、Memory Bank构建、Coreset采样
- 预测和模型保存/加载

### 3. data/dataset.py
数据集加载器：
- `MVTecDataset` 类：处理比赛数据集格式
- 支持 train/threshold/test 三种数据划分

### 4. utils/metrics.py
评估指标相关：
- `search_best_threshold()`: 搜索最佳阈值（带过拟合检测）
- `compute_metrics()`: 计算F1等指标

### 5. utils/visualize.py
可视化功能：
- `anomaly_map_to_heatmap()`: 生成热力图
- `overlay_heatmap_on_image()`: 叠加热力图
- `find_anomaly_regions()`: 检测异常区域
- `draw_boxes_on_image()`: 绘制边界框
- `visualize_anomaly_detection()`: 完整可视化流程

## 使用方式

### 训练
```bash
python code/train.py
```

### 评估
```bash
python code/inference.py
```

### 可视化
```bash
python code/visualize_example.py
```

## 导入方式

所有模块都使用标准包导入：

```python
from code.data import MVTecDataset
from code.models import PatchCore
from code.utils import search_best_threshold, visualize_anomaly_detection
from code.config import DATASET_ROOT, MODEL_CONFIG, TRAIN_CONFIG
```

## 优势

1. **模块化**：功能清晰分离，易于维护
2. **配置集中**：所有参数在config.py统一管理
3. **标准结构**：符合Python包结构规范
4. **易于扩展**：添加新功能只需在对应模块添加

