import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 创建目录用于保存数据
if not os.path.exists('mnist_augmented'):
    os.makedirs('mnist_augmented')

# 下载MNIST数据集
(_, _), (test_images, test_labels) = mnist.load_data()

# 选择测试集中的>100张图片
selected_images = test_images[:200]
selected_labels = test_labels[:200]

# 数据增强到100张
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')
'''
rotation_range：整数，数据提升时图片随机转动的角度。随机选择图片的角度，是一个0~180的度数，取值为0~180。 在 [0, 指定角度] 范围内进行随机角度旋转。
width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度。
height_shift_range：浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。 height_shift_range和width_shift_range是用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。
shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度。
zoom_range：当给出一个数时，图片同时在长宽两个方向进行同等程度的放缩操作；当给出一个list时，则代表[width_zoom_range, height_zoom_range]，即分别对长宽进行不同程度的放缩。
            而参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作。
channel_shift_range：浮点数，随机通道偏移的幅度。
horizontal_flip：布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候。
vertical_flip：布尔值，进行随机竖直翻转。
fill_mode：'constant'，'nearest'，'reflect'或'wrap'之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
'''

# 将选取的图片进行数据增强
augmented_images = []
augmented_labels = []
for i, (image, label) in enumerate(zip(selected_images, selected_labels)):
    image = np.expand_dims(image, axis=2)
    augmented_image_set = datagen.flow(np.expand_dims(image, axis=0), batch_size=5)
    augmented_images.extend([np.squeeze(aug_img, axis=2) for aug_img in augmented_image_set[0]])
    augmented_labels.extend([label] * 5)

# 将增强后的图片和标签写入文件
with open('mnist_augmented/mnist_augmented.txt', 'w') as f:
    for i, (image, label) in enumerate(zip(augmented_images, augmented_labels)):
        filename = f'mnist_augmented/{i}.png'
        # 保存图像文件
        plt.imsave(filename, image, cmap='gray')
        # 写入文件路径和标签
        f.write(f'{filename} {label}\n')
print("数据增强并写入文件完成。")


# 加载模型
model = torch.load("models/module_20.pth")
device = torch.device("cpu")

# 准备写入结果的文件
result_file = open("evaluation_test.txt", "w")

# 在测试集上评测模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    count = 0
    for images, labels in enumerate(zip(augmented_images, augmented_labels)):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 写入评测结果到文件
        result_file.write(f"Image {count + 1}: Predicted {predicted.item()}, Ground Truth {labels.item()}\n")

        count += 1
        if count >= 100:
            break
# 关闭文件
result_file.close()

# 打印准确率
accuracy = 100 * correct / total
print('准确率: {} %'.format(accuracy))


