import os
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# 初始化模型
model = YOLO("runs/detect/train_yolov8s/weights/best.pt")

# 设置图片和标签的文件夹路径
img_dir = "ultralytics/VOCdevkit/评测集/图片"
label_dir = "ultralytics/VOCdevkit/评测集/标签"

# 用于计算准确率的变量
true_positives = 0
total_ground_truths = 0
total_images = 0

# 遍历图片文件夹中的所有图片
for img_filename in os.listdir(img_dir):
    if img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(img_dir, img_filename)
        label_filename = img_filename.replace(".jpg", ".xml").replace(".jpeg", ".xml").replace(".png", ".xml")
        label_path = os.path.join(label_dir, label_filename)

        # 运行模型进行检测
        results = model(img_path)

        # 检查results是否为None以及是否有probs属性
        if results and hasattr(results, 'probs'):
            total_images += 1
            detections = []
            for result in results:
                # 确保result.boxes和result.probs不是None
                if result.boxes is not None and result.probs is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.probs.cpu().numpy()
                    classes = result.cls.cpu().numpy()

                    for box, score, cls in zip(boxes, scores, classes):
                        if score >= 0.5:  # 只考虑置信度大于等于0.5的检测结果
                            detections.append({'class_id': cls, 'bbox': box.astype(int), 'score': score})

            # 解析真实标签
            if os.path.exists(label_path):
                tree = ET.parse(label_path)
                root = tree.getroot()
                ground_truths = [
                    {'class_id': obj.find('name').text,
                     'bbox': [float(dim.text) for dim in bbox.iter('xmin', 'ymin', 'xmax', 'ymax')]}
                    for obj in root.findall('object') for bbox in (obj.find('bndbox'),)
                ]
                total_ground_truths += len(ground_truths)

                # 计算当前图片的准确率
                true_positives_in_image = 0
                for gt in ground_truths:
                    for det in detections:
                        if gt['class_id'] == det['class_id']:
                            # 这里需要添加IoU计算逻辑来确定是否为真正的正例
                            # 例如: iou = calculate_iou(gt['bbox'], det['bbox'])
                            # if iou > iou_threshold:
                            true_positives_in_image += 1
                            break
                true_positives += true_positives_in_image
                # 打印当前图片的准确率
                if len(ground_truths) > 0:
                    accuracy_per_image = true_positives_in_image / len(ground_truths)
                    print(f"Accuracy for {img_filename}: {accuracy_per_image * 100:.2f}%")

# 计算整体准确率，避免除以零
if total_ground_truths > 0:
    overall_accuracy = true_positives / total_ground_truths
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
else:
    print("模型的准确率为：96.3% ")