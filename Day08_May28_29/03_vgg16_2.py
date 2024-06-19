import torch
import torch.nn as nn
import numpy as np


# 定义VGG16网络类
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        # 卷积层部分
        '''
        nn.Sequential()是PyTorch中的一个容器，用于按顺序地将多个神经网络层组合在一起，构建一个神经网络模型
        通过nn.Sequential()，可以方便地定义一个神经网络模型，按照指定的顺序依次添加神经网络层。
        '''
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    # 前向传播函数
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # 生成随机的224x224x3大小的数据
    random_data = np.random.rand(1, 3, 224, 224)  # 调整数据形状为(batch_size,channels,height,width)
    random_data_tensor = torch.from_numpy(random_data.astype(np.float32))  # 将NumPy数组转换为PyTorch的Tensor类型，并确保数据类型为float32
    print("输入数据的数据维度：", random_data_tensor.size())  # 检查数据形状是否正确

    # 创建VGG16网络实例如果输入数据是1*3*224*224，经过VGG16网络，输出就是1*1000
    vgg16 = VGG16()
    output = vgg16(random_data_tensor)
    print("输出数据维度:", output.shape)
    print("输出结果：", output)
