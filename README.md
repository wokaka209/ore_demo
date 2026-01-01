# 矿石粒径分析系统

基于YOLOv11的矿石粒径分析与测量系统，用于自动化检测和分析矿石颗粒的尺寸分布。

## 项目简介

本项目使用YOLOv11分割模型对矿石图像进行精确分割和粒径分析。系统能够自动检测图像中的矿石颗粒，计算其尺寸参数，并生成相应的统计图表。

## 功能特性

- **精确分割**：使用YOLOv11-seg模型对矿石颗粒进行像素级分割
- **尺寸测量**：计算检测到的矿石颗粒的尺寸参数
- **批量处理**：支持批量图像预测和分析
- **结果可视化**：生成统计图表展示粒径分布
- **自适应训练**：根据GPU内存自动调整训练参数

## 技术栈

- Python 3.9+
- PyTorch
- YOLOv11 (Ultralytics)
- OpenCV
- NumPy
- Matplotlib
- Pandas

## 项目结构

```
ore_size_grain/
├── my_train.py          # 模型训练脚本
├── my_prediction.py     # 模型预测脚本
├── json2yolo.py         # JSON标注转YOLO格式
├── yolo_dataset_seg/    # YOLO分割格式数据集
├── runs/                # 训练结果保存目录
└── README.md
```


## 数据准备

数据集需要按照YOLO格式组织：

```
yolo_dataset_seg/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

## 使用方法

### 1. 训练模型

```bash
python my_train.py
```



### 2. 进行预测

```bash
python my_prediction.py
```



### 3. 数据格式转换

如果使用LabelMe标注的数据，可以使用以下脚本转换格式：

```bash
python datasets2yolo.py
```

## 训练参数说明

- **图像尺寸**：640x640
- **批次大小**：根据GPU内存自适应（8GB显存使用4）
- **训练轮数**：100轮
- **数据增强**：启用随机增强
- **学习率**：自适应调整

## 模型性能

模型在验证集上的表现：
- 检测mAP50: ~0.99
- 分割mAP50: ~0.99
- 检测mAP50-95: ~0.89
- 分割mAP50-95: ~0.77

## 结果输出

预测结果包括：
- 带有分割掩码的图像
- 检测框坐标
- 掩码轮廓信息
- 尺寸测量数据

## 自定义配置

你可以在训练脚本中调整以下参数：
- `epochs`: 训练轮数
- `imgsz`: 图像尺寸
- `batch`: 批次大小
- `device`: 训练设备
- `project`: 项目保存路径
- `name`: 训练任务名称

