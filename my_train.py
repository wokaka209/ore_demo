import os
import torch
from ultralytics import YOLO
import yaml
import pandas as pd

def get_optimal_batch_size():
    """
    根据GPU内存自动确定最佳批次大小
    """
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory >= 11:
            return 32
        elif gpu_memory >= 8:
            return 4
        elif gpu_memory >= 6:
            return 4
        else:
            return 2
    else:
        # CPU训练时使用较小的批次
        return 2

def train_model():
    """
    训练YOLO模型
    """
    print("开始训练YOLOv11分割模型...")
    
    # 检查是否有GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device if device != 'cpu' else 'CPU'}")
    
    # 获取最佳批次大小
    batch_size = 16
    print(f"使用批次大小: {batch_size}")

    my_bestpt = 'D:/Graduate student tasks/ore_size_grain/runs/train/ore_seg_training_v2/weights/best.pt'
    # 加载预训练模型
    model = YOLO('yolo11n-seg.pt')
    
    # 训练参数
    epochs = 150  # 增加训练轮数以进一步降低seg_loss
    
    # 开始训练
    results = model.train(
        data='yolo_dataset_seg/dataset.yaml',  # 数据集配置文件
        epochs=epochs,
        imgsz=640,  # 图像尺寸
        batch=batch_size,  # 批次大小
        device=device,
        project='runs/train',  # 项目路径
        name='ore_seg_training_v3',  # 新的训练运行名称
        save=True,  # 保存模型
        verbose=True,
        patience=0,  # 早停轮数
        fraction=0.9,  # 使用90%的数据作为训练集
        exist_ok=True,  # 允许覆盖已存在的训练结果
        workers=8,
        plots=False,  # 禁用绘图以避免PR曲线错误
        # 调整损失权重以特别关注分割损失
        box=7.5,      # 框损失权重
        cls=0.5,      # 分类损失权重
        dfl=1.5,      # 分布焦点损失权重
        # 尝试更小的学习率以获得更精细的收敛
        lr0=0.005,    # 初始学习率
        lrf=0.005     # 最终学习率
    )
    
    print("训练完成！")
    print(f"最佳权重保存在: {results.save_dir}/weights/best.pt")
    
    # 检查并复制结果CSV文件
    import glob
    result_csv_files = glob.glob(os.path.join(results.save_dir, 'results.csv'))
    if result_csv_files:
        csv_path = result_csv_files[0]
        output_csv = os.path.join(results.save_dir, f'training_results.csv')
        import shutil
        shutil.copy2(csv_path, output_csv)
        print(f"训练结果CSV文件已保存至: {output_csv}")
    else:
        print("警告: 未找到训练结果CSV文件")
    
    return results

if __name__ == "__main__":
    train_model()