import os
import numpy as np
import matplotlib.pyplot as plt

# 设置路径
img_train_dir = 'VOCdevkit/images/train/'
img_val_dir = 'VOCdevkit/images/val/'
label_train_dir = 'VOCdevkit/labels/train/'
label_val_dir = 'VOCdevkit/labels/val/'
train_list_file = 'yolov8_train.txt'
val_list_file = 'yolov8_val.txt'

# 统计每个类别的图片和检测框数量
def count_objects(label_dir, img_dir):
    labels = os.listdir(label_dir)
    label_counts = {}
    box_counts = {}
    total_img_count = 0
    total_box_count = 0
    mismatched_files = []

    for label_file in labels:
        with open(os.path.join(label_dir, label_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                items = line.split()
                label = items[0]
                box_counts[label] = box_counts.get(label, 0) + 1
                total_box_count += 1

            label_counts[label_file.split('.')[0]] = len(lines)

    # 检查图片文件是否一一对应
    img_files = os.listdir(img_dir)
    for img_file in img_files:
        if img_file.split('.')[0] not in label_counts:
            mismatched_files.append(img_file)

    img_count = len(img_files)
    total_img_count += img_count

    return label_counts, box_counts, total_img_count, total_box_count, mismatched_files

def visualize_histogram(data, title, xlabel, ylabel):
    x = np.arange(len(data))
    labels = list(data.keys())
    values = list(data.values())

    plt.bar(x, values, align='center', alpha=0.5)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

# 统计训练集数据
train_label_counts, train_box_counts, train_total_img_count, train_total_box_count, train_mismatched_files = count_objects(label_train_dir, img_train_dir)

# 绘制训练集直方图
visualize_histogram(train_label_counts, 'Number of Images per Class (Train)', 'Class', 'Number of Images')
visualize_histogram(train_box_counts, 'Number of Boxes per Class (Train)', 'Class', 'Number of Boxes')

# 统计验证集数据
val_label_counts, val_box_counts, val_total_img_count, val_total_box_count, val_mismatched_files = count_objects(label_val_dir, img_val_dir)

# 绘制验证集直方图
visualize_histogram(val_label_counts, 'Number of Images per Class (Val)', 'Class', 'Number of Images')
visualize_histogram(val_box_counts, 'Number of Boxes per Class (Val)', 'Class', 'Number of Boxes')

# 输出数据总结
print('Train dataset summary:')
print('Total number of images:', train_total_img_count)
print('Total number of boxes:', train_total_box_count)
print('Average boxes per image:', train_total_box_count / train_total_img_count)
print('Mismatched files:', train_mismatched_files)

print('Val dataset summary:')
print('Total number of images:', val_total_img_count)
print('Total number of boxes:', val_total_box_count)
print('Average boxes per image:', val_total_box_count / val_total_img_count)
print('Mismatched files:', val_mismatched_files)