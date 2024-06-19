import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 搭建LeNet-1网络模型
class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # 两个2d卷积层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 一个dropout层
        self.conv2_drop = nn.Dropout2d()
        # 全连接层
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # 前向传播函数
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 输入层
        x = x.view(-1, 320)
        # 隐藏层
        x = self.fc1(x)
        # 隐藏层的激活函数
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # 输出层
        x = self.fc2(x)
        # 输出层的激活函数，返回softmax的计算值
        return F.log_softmax(x, dim=1)