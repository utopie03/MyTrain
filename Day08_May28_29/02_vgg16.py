import torch
import torch.nn as nn
import numpy as np


# 定义VG616网络类
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 卷积层部分
        # nn.Conv2d参数说明: in_channels:输入特征图的通道数  out_channels:输出特征图的通道数
        # kernel_size:卷积核的大小 stride:卷积运算的步幅 padding:前向计算时在输入特征图周围添加的像素数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # inplace:是否改变输入数据
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu13 = nn.ReLU(inplace=True)
        self.max_pooling5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层部分
        # nn.Linear参数说明: in_features:输入神经元个数 out_features:输出神经元个数 bias=TUe:是否包含偏置
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu14 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu15 = nn.ReLU(inplace=True)
        # nn.Dropout默认取值0.5
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(4096, 1000)

    # 前向传播函数
    def forward(self, x):
        # 卷积层1
        x = self.conv1(x)
        # 激活函数
        x = self.relu1(x)
        # 卷积层2
        x = self.conv2(x)
        # 激活函数
        x = self.relu2(x)
        # 最大池化层1
        x = self.max_pooling1(x)

        # 卷积层3
        x = self.conv3(x)
        # 激活函数
        x = self.relu3(x)
        # 卷积层4
        x = self.conv4(x)
        # 激活函数
        x = self.relu4(x)
        # 最大池化层2
        x = self.max_pooling2(x)

        # 卷积层5
        x = self.conv5(x)
        # 激活函数
        x = self.relu5(x)
        # 卷积层6
        x = self.conv6(x)
        # 激活函数
        x = self.relu6(x)
        # 卷积层7
        x = self.conv7(x)
        # 激活函数
        x = self.relu7(x)
        # 最大池化层3
        x = self.max_pooling3(x)

        # 卷积层8
        x = self.conv8(x)
        # 激活函数
        x = self.relu8(x)
        # 卷积层9
        x = self.conv9(x)
        # 激活函数
        x = self.relu9(x)
        # 卷积层10
        x = self.conv10(x)
        # 激活函数
        x = self.relu10(x)
        # 最大池化层4
        x = self.max_pooling4(x)

        # 卷积层11
        x = self.conv11(x)
        # 激活函数ReLu11
        x = self.relu11(x)
        # 卷积层12
        x = self.conv12(x)
        # 激活函数
        x = self.relu12(x)
        # 卷积层13
        x = self.conv13(x)
        # 激活函数
        x = self.relu13(x)
        # 最大池化层5
        x = self.max_pooling5(x)
        print("x shape111: ", x.shape)

        # 将x的形状重新调整为[-1，512*7*7] 即[1，25088]
        # 当你使用.view()函数对张量进行形状变换时，可以用-1来表示一个特殊的值，
        # 它表示该维度的大小由函数自动推嘶而来，以保证张量的总元素数不变
        x = x.view(-1, 512 * 7 * 7)
        print("x shape222: ", x.shape)
        # 全连接层fc1 1*25088->1*4096
        x = self.fc1(x)
        # 激活函数
        x = self.relu14(x)
        # 全连接层fc2 2*4096->1*4096
        x = self.fc2(x)
        # 激活函数
        x = self.relu15(x)
        # 全连接层fc3 1*4096->1*1000
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # 生成随机的224x224x3大小的数据 BCHMN BHWC opencv: HWC(512，512，3)
    random_data = np.random.rand(1, 3, 224, 224)  # 调整数据形状为(batch_size,channels,height,width)
    random_data_tensor = torch.from_numpy(random_data.astype(np.float32))  # 将NumPy数组转换为PyTorch的Tensor类型，并确保数据类型为float32
    print("输入数据的数据维度：", random_data_tensor.size())  # 检查数据形状是否正确

    # 创建VGG16网络实例  如果输入数据是1*3*224*224，经过VGG16网络，输出就是1*1000
    vgg16 = VGG16()
    output = vgg16(random_data_tensor)
    print("输出数据维度:", output.shape)
    print("输出结果：", output)