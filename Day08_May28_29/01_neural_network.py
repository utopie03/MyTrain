import torch
import torch.nn as nn
import torch.optim as optim

# import torch.nn.functional as E

'''
pip install torch==1.13.1 torchvision==0.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
ERROR: Could not find a version that satisfies the requirement torch==1.13.1 (from versions:
 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.5.0, 1.5.1, 1.6.0, 1.7.0, 1.7.1, 1.8.0,
 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2)
ERROR: Could not find a version that satisfies the requirement torchvision==0.14.1 
(from versions: 0.1.6, 0.1.7, 0.1.8, 0.1.9, 0.2.0, 0.2.1, 0.2.2, 0.2.2.post2, 0.2.2.post3,
 0.3.0, 0.4.0, 0.4.1, 0.4.2, 0.5.0, 0.6.0, 0.6.1, 0.7.0, 0.8.0, 0.8.1, 0.8.2, 0.9.0, 0.9.1, 
 0.10.0, 0.10.1, 0.11.0, 0.11.1, 0.11.2)
'''


# 定义一个简单的神经网络类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 输入大小10，输出大小5的全连接层
        self.fc2 = nn.Linear(5, 2)  # 输入大小5，输出大小2的全连接层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 应用ReLU激活函数
        x = self.fc2(x)  # 直接输出，不再应用激活函数
        return x


# 实例化神经网络
model = NeuralNetwork()
print("model: ", model)

# 随机生成一些数据
x = torch.randn(1, 10)  # 输入数据，大小为1x10
y = torch.randn(1, 2)  # 期望的输出数据，大小为1x2

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 前向传播、计算损失、反向传播、优化参数
optimizer.zero_grad()  # 清零梯度
outputs = model(x)  # 前向传播
loss = criterion(outputs, y)  # 计算损失
loss.backward()  # 反向传播
optimizer.step()  # 优化参数

# 输出网络参数
print(model.fc1.weight)
