"""
将LabelMe格式的JSON标注文件转换为YOLO分割格式，并按8:1:1比例划分数据集
用于YOLOv11-seg等分割模型的训练
"""
import json
import os
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple


def normalize_polygon_points(points: List[List[float]], img_width: int, img_height: int) -> List[float]:
    """
    将LabelMe格式的多边形点坐标转换为YOLO分割格式的归一化坐标
    
    Args:
        points: 多边形的点坐标列表，格式为[[x1, y1], [x2, y2], ...]
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        归一化后的点坐标列表 [x1, y1, x2, y2, ...]
    """
    if not points:
        return []
    
    normalized_points = []
    for point in points:
        x, y = point
        normalized_x = x / img_width
        normalized_y = y / img_height
        normalized_points.extend([normalized_x, normalized_y])
    
    return normalized_points


def process_json_file(json_path: str, output_dir: str) -> None:
    """
    处理单个JSON文件，将其转换为YOLO分割格式
    
    Args:
        json_path: JSON文件路径
        output_dir: 输出目录
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图像尺寸
    img_width = data.get('imageWidth')
    img_height = data.get('imageHeight')
    
    if not img_width or not img_height:
        print(f"警告: 无法从 {json_path} 获取图像尺寸")
        return
    
    # 获取标注形状
    shapes = data.get('shapes', [])
    
    yolo_annotations = []
    for shape in shapes:
        label = shape.get('label', 'unknown')
        points = shape.get('points', [])
        
        # 转换为YOLO分割格式
        normalized_points = normalize_polygon_points(points, img_width, img_height)
        
        if normalized_points:
            # 假设标签为0（如果需要多个类别，请根据实际情况调整）
            # 这里假设所有标注都是同一类，类别索引为0
            # 如果有多个类别，需要建立类别映射
            class_id = 0  # 默认类别ID，可根据需要修改
            points_str = ' '.join([f"{coord:.6f}" for coord in normalized_points])
            yolo_annotations.append(f"{class_id} {points_str}")
    
    # 生成输出文件名（与JSON同名，扩展名为.txt）
    output_filename = Path(json_path).stem + '.txt'
    output_path = os.path.join(output_dir, output_filename)
    
    # 写入YOLO格式文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for annotation in yolo_annotations:
            f.write(annotation + '\n')


def create_dataset_split(image_paths: List[str], split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[List[str], List[str], List[str]]:
    """
    按照给定比例划分数据集
    
    Args:
        image_paths: 图像路径列表
        split_ratios: 划分比例 (train, val, test)
    
    Returns:
        划分后的路径列表 (train_paths, val_paths, test_paths)
    """
    # 随机打乱路径列表
    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)
    
    # 计算划分边界
    total_count = len(shuffled_paths)
    train_count = int(total_count * split_ratios[0])
    val_count = int(total_count * split_ratios[1])
    
    # 划分数据集
    train_paths = shuffled_paths[:train_count]
    val_paths = shuffled_paths[train_count:train_count + val_count]
    test_paths = shuffled_paths[train_count + val_count:]
    
    return train_paths, val_paths, test_paths


def create_yolo_dataset(input_dir: str, output_dir: str, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    创建YOLO格式的数据集
    
    Args:
        input_dir: 输入目录，包含图像和JSON标注文件
        output_dir: 输出目录
        split_ratios: 划分比例 (train, val, test)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录结构
    train_img_dir = output_path / 'images' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    test_img_dir = output_path / 'images' / 'test'
    
    train_label_dir = output_path / 'labels' / 'train'
    val_label_dir = output_path / 'labels' / 'val'
    test_label_dir = output_path / 'labels' / 'test'
    
    for dir_path in [train_img_dir, val_img_dir, test_img_dir,
                     train_label_dir, val_label_dir, test_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(input_path.glob(f'*{ext}')))
        image_paths.extend(list(input_path.glob(f'*{ext.upper()}')))
    
    # 按比例划分数据集
    train_paths, val_paths, test_paths = create_dataset_split(image_paths, split_ratios)
    
    print(f"数据集划分完成:")
    print(f"训练集: {len(train_paths)} 个文件")
    print(f"验证集: {len(val_paths)} 个文件")
    print(f"测试集: {len(test_paths)} 个文件")
    
    # 处理每个子集
    for subset_name, subset_paths in [('train', train_paths), ('val', val_paths), ('test', test_paths)]:
        print(f"\n处理 {subset_name} 子集...")
        
        # 获取对应的图像和标签目录
        img_dir = output_path / 'images' / subset_name
        label_dir = output_path / 'labels' / subset_name
        
        for img_path in subset_paths:
            # 复制图像文件
            dest_img_path = img_dir / img_path.name
            shutil.copy2(img_path, dest_img_path)
            
            # 查找对应的JSON文件
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                # 转换JSON标注为YOLO格式
                process_json_file(str(json_path), str(label_dir))
            else:
                print(f"警告: 未找到 {img_path} 对应的JSON标注文件")
    
    # 创建dataset.yaml配置文件
    create_dataset_yaml(output_path, train_img_dir, val_img_dir, test_img_dir, 
                       train_label_dir, val_label_dir, test_label_dir)


def create_dataset_yaml(output_dir: Path, train_img_dir: Path, val_img_dir: Path, test_img_dir: Path,
                       train_label_dir: Path, val_label_dir: Path, test_label_dir: Path):
    """
    创建YOLO分割数据集配置文件
    """
    yaml_content = f"""# YOLO数据集配置文件
path: {output_dir.absolute()}  # 数据集根目录
train: {os.path.relpath(train_img_dir, output_dir)}  # 训练图像目录
val: {os.path.relpath(val_img_dir, output_dir)}  # 验证图像目录
test: {os.path.relpath(test_img_dir, output_dir)}  # 测试图像目录 (可选)

# 类别
nc: 1  # 类别数量
names: ['ore']  # 类别名称列表

# 训练时使用的标签目录
train_labels: {os.path.relpath(train_label_dir, output_dir)}
val_labels: {os.path.relpath(val_label_dir, output_dir)}
test_labels: {os.path.relpath(test_label_dir, output_dir)}
"""
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n数据集配置文件已创建: {yaml_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='将JSON标注转换为YOLO格式数据集')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录路径，包含图像和JSON标注文件')
    parser.add_argument('--output_dir', type=str, default='./yolo_dataset',
                        help='输出目录路径 (默认: ./yolo_dataset)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例 (默认: 0.1)')
    
    args = parser.parse_args()
    
    # 验证比例和
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"错误: 比例之和必须为1.0，当前为 {total_ratio}")
        return
    
    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    
    print(f"开始转换数据集...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分比例 - 训练集: {args.train_ratio}, 验证集: {args.val_ratio}, 测试集: {args.test_ratio}")
    
    create_yolo_dataset(args.input_dir, args.output_dir, split_ratios)
    
    print(f"\n数据集转换完成！")


if __name__ == "__main__":
    main()