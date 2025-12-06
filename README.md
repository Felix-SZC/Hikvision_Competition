# Hikvision_Competition
第四届启智杯机器智能大赛-视觉算法创新赛道

## 功能特性

### 异常检测
- 基于PatchCore算法的工业缺陷检测
- 支持10个不同类别的产品检测
- 高精度异常识别和分类

### 可视化功能 ✨ **新增**
- **热力图显示**: 将异常区域以彩色热力图形式叠加在原图上
- **智能框选**: 自动检测并框出异常区域，支持配置敏感度
- **详细信息标注**: 显示异常分数、阈值、状态和检测到的异常区域数量

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
cd code
python train.py
```

### 运行推理和可视化
```bash
# 启用可视化（修改code/config.py中的visualize=True）
python inference.py
```

### 查看演示
```bash
python demo_visualization.py
```

## 可视化配置

在 `code/config.py` 中调整参数：
```python
EVAL_CONFIG = {
    'visualize': True,           # 是否生成可视化
    'max_vis_samples': 20,       # 每个类别生成图片数，-1=全部
}

VISUALIZATION_CONFIG = {
    'red_threshold': 200,        # 框出敏感度，越高越精确
}
```

可视化结果保存在 `visualizations/` 目录中。