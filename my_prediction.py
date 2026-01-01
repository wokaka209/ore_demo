import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import math
from pathlib import Path
import yaml

def calculate_particle_info(mask):
    """
    计算颗粒的粒径信息
    """
    # 寻找轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, None
    
    # 获取最大轮廓（假设是颗粒）
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 10:  # 过滤小面积噪声
        return None, None, None, None
    
    # 拟合椭圆
    if len(largest_contour) >= 5:  # 至少需要5个点才能拟合椭圆
        ellipse = cv2.fitEllipse(largest_contour)
        center, axes, angle = ellipse
        major_axis = max(axes)  # 长轴
        minor_axis = min(axes)  # 短轴
        # 计算等效直径（基于面积）
        area = cv2.contourArea(largest_contour)
        equivalent_diameter = 2 * math.sqrt(area / math.pi)
    else:
        # 如果轮廓点太少，使用边界矩形估算
        x, y, w, h = cv2.boundingRect(largest_contour)
        major_axis = max(w, h)
        minor_axis = min(w, h)
        area = cv2.contourArea(largest_contour)
        equivalent_diameter = 2 * math.sqrt(area / math.pi)
        center = (x + w/2, y + h/2)
    
    return center, equivalent_diameter, major_axis, minor_axis

def predict_with_model():
    """
    使用训练好的模型进行预测
    """
    # 检查输出目录是否存在，不存在则创建
    output_dir = Path("output/predict_testdata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载训练好的模型
    # 首先尝试加载最新的best.pt，如果不存在则使用默认路径
    model_path = "runs/train/ore_seg_training_v2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"未找到训练好的模型 {model_path}，请先运行训练程序")
        return
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 获取测试数据集中的图片
    dataset_yaml = "yolo_dataset_seg/dataset.yaml"
    
    # 读取数据集配置文件以获取测试集路径
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    test_path = dataset_config.get('test')
    if not test_path:
        print("无法找到测试数据集路径")
        return
    
    # 获取测试图片列表
    test_images_dir = os.path.join(os.path.dirname(dataset_yaml), test_path)
    if not os.path.exists(test_images_dir):
        print(f"测试数据集路径不存在: {test_images_dir}")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    test_images = []
    for ext in image_extensions:
        test_images.extend(Path(test_images_dir).glob(f"*{ext}"))
        test_images.extend(Path(test_images_dir).glob(f"*{ext.upper()}"))
    
    if not test_images:
        print(f"在 {test_images_dir} 中未找到测试图片")
        return
    
    print(f"找到 {len(test_images)} 张测试图片")
    
    # 对每张测试图片进行预测
    for i, img_path in enumerate(test_images):
        print(f"处理图片 {i+1}/{len(test_images)}: {img_path.name}")
        
        # 进行预测
        results = model(str(img_path))
        
        # 读取原图
        original_img = cv2.imread(str(img_path))
        if original_img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        # 复制原图用于绘制结果
        result_img = original_img.copy()
        
        # 获取分割掩码、边界框和类别信息
        if results and len(results) > 0:
            result = results[0]
            
            if result.masks is not None:  # 分割任务
                masks = result.masks.data.cpu().numpy()  # 掩码
                boxes = result.boxes  # 边界框
                names = result.names  # 类别名称
                
                # 存储所有颗粒的粒径信息用于统计
                all_particles_info = []
                
                # 遍历每个检测到的对象
                for j, (mask,box) in enumerate(zip(masks,boxes)):
                    # 调整掩码大小以匹配原图
                    mask_resized = cv2.resize(mask.astype(np.uint8), (original_img.shape[1], original_img.shape[0]))
                    
                    # 创建彩色掩码
                    color = np.random.randint(0, 255, 3).tolist()  # 随机颜色
                    mask_colored = np.zeros_like(result_img)
                    mask_colored[mask_resized == 1] = color
                    
                    # 将掩码叠加到原图上（半透明）
                    result_img = cv2.addWeighted(result_img, 1.0, mask_colored, 0.5, 0)
                    
                    # 计算颗粒信息
                    center, diameter, major_axis, minor_axis = calculate_particle_info(mask_resized)
                    
                    if center is not None:
                        # 在图上绘制颗粒信息
                        x, y = int(center[0]), int(center[1])
                        
                        # 绘制颗粒ID
                        cv2.putText(result_img, f'ID:{j+1}', (x, y-25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # 绘制粒径信息
                        info_text = f'Dia:{diameter:.1f}'
                        cv2.putText(result_img, info_text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 绘制长短轴信息
                        axes_text = f'L:{major_axis:.1f}, S:{minor_axis:.1f}'
                        cv2.putText(result_img, axes_text, (x, y+10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # 保存颗粒信息用于后续统计
                        all_particles_info.append({
                            'id': j+1,
                            'diameter': diameter,
                            'major_axis': major_axis,
                            'minor_axis': minor_axis
                        })
                        
                        # # 绘制椭圆（如果可用）
                        # if major_axis is not None and minor_axis is not None:
                        #     # 这里简化处理，绘制一个矩形来表示颗粒大小
                        #     size_factor = 1.0  # 可调整大小显示比例
                        #     pt1 = (int(x - minor_axis/2 * size_factor), int(y - major_axis/2 * size_factor))
                        #     pt2 = (int(x + minor_axis/2 * size_factor), int(y + major_axis/2 * size_factor))
                        #     cv2.rectangle(result_img, pt1, pt2, (0, 255, 0), 1)
        
        # 绘制粒径分布信息到图片右下角
        if all_particles_info:
            # 计算不同粒径范围的颗粒数量
            diameters = [p['diameter'] for p in all_particles_info]
            
            # 按粒径范围统计
            range_0_20 = sum(1 for d in diameters if 0 <= d <= 20)
            range_21_40 = sum(1 for d in diameters if 21 <= d <= 40)
            range_41_60 = sum(1 for d in diameters if 41 <= d <= 60)
            range_61_80 = sum(1 for d in diameters if 61 <= d <= 80)
            range_81_100 = sum(1 for d in diameters if 81 <= d <= 100)
            range_100_plus = sum(1 for d in diameters if d > 100)
            total_particles = len(diameters)
            
            # 设置文本参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # 计算文本大小和位置
            stats_text = [
                f'0_20_dia:{range_0_20}',
                f'21_40_dia:{range_21_40}',
                f'41_60_dia:{range_41_60}',
                f'61_80_dia:{range_61_80}',
                f'81_100_dia:{range_81_100}',
                f'100_plus_dia:{range_100_plus}',
                f'total_ore_num:{total_particles}'
            ]
            
            # 获取图片尺寸
            img_height, img_width = result_img.shape[:2]
            
            # 设置右下角的起始位置
            margin = 10
            line_height = 20
            start_y = img_height - (len(stats_text) * line_height + margin)
            
            # 绘制背景矩形
            bg_start_x = img_width - 200
            bg_start_y = start_y - 5
            bg_end_x = img_width - margin
            bg_end_y = start_y + len(stats_text) * line_height + 5
            cv2.rectangle(result_img, (bg_start_x, bg_start_y), (bg_end_x, bg_end_y), (255, 255, 255), -1)
            
            # 绘制统计信息
            for idx, text in enumerate(stats_text):
                y_pos = start_y + idx * line_height
                cv2.putText(result_img, text, (img_width - 190, y_pos), 
                            fontFace=font, fontScale=font_scale, color=(0, 0, 255), thickness=2)
        
        # 保存结果图片
        output_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), result_img)
        print(f"结果已保存: {output_path}")
    
    print(f"预测完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    predict_with_model()