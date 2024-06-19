import cv2
import matplotlib.pyplot as plt
import torch
import torchvision

from Net import *  # 网络模型文件
from torch.utils.data import DataLoader

test_data = torchvision.datasets.MNIST("./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.ToTensor()
])

# 选择测试集的第二个图片进行推理
image, label = test_data[1]
PIL_Image = torchvision.transforms.ToPILImage()

pic = PIL_Image(image)
# 保存图片
pic.save("random.jpg")

# 模型的权重选择
model = torch.load("models/module_20.pth")
device = torch.device("cpu")
image = image.to(device)

image = torch.reshape(image, (1, 1, 28, 28))
model.eval()
with torch.no_grad():
    output = model(image)
print("预测的结果为：{}".format(output.argmax(1).item()))

# 可视化推理结果
# 读取图片
random_img = cv2.imread("random.jpg")
random_img = cv2.cvtColor(random_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5, 5))
plt.title('Input Image', fontsize=16)

# 隐藏坐标轴刻度
plt.xticks([]), plt.yticks([])
plt.xlabel('The predicted result is: {}'.format(output.argmax(1).item()), fontsize=15)
# plt.axis('off')
'''
# 图片上添加文字
plt.text(0.1, 0.9,  # 坐标
         'The predicted result is: {}'.format(output.argmax(1).item()),  # 文字
         weight='black',  # 字体，这里是黑体
         color='w'  # 颜色：白色
         )
'''
plt.imshow(random_img)
plt.show()
