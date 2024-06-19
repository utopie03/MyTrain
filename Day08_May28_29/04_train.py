import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from LeNet import *  # 网络模型文件
from torch import nn
from torch.utils.data import DataLoader


# pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple/
def train():
    train_data = torchvision.datasets.CIFAR10(root="CIFAR10",
                                              train=True,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)
    test_data = torchvision.datasets.CIFAR10(root="CIFAR10",
                                             train=False,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)

    # 利用DataLoader来加载数据集
    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # 创建网络模型
    module = Module()
    if torch.cuda.is_available():
        module = module.cuda()
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    # 优化器
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)

    # 训练的轮数
    epoch = 20
    # 储存路径
    work_dir = './LeNet'
    # 添加tensorboard
    writer = SummaryWriter("{}/logs".format(work_dir))

    for i in range(epoch):
        print("-------epoch {}-------".format(i + 1))
        # 训练步骤
        module.train()
        for step, [imgs, targets] in enumerate(train_dataloader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = module(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step = len(train_dataloader) * i + step + 1
            if train_step % 100 == 0:
                print("train time: {}，Loss: {}".format(train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), train_step)

        # 测试步骤
        module.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for imgs, targets in test_dataloader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                outputs = module(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()  # argmax(1)表示把outputs矩阵中的最大值输出
                total_accuracy = total_accuracy + accuracy

        print("test set loss:{}".format(total_test_loss))
        print("test set accuracy:{}".format(total_accuracy / len(test_data)))
        writer.add_scalar("test_loss", total_test_loss, i)
        writer.add_scalar("test_accuracy", total_accuracy / len(test_data), i)

        torch.save(module, "{}/module_{}.pth".format(work_dir, i + 1))
        print("saved epoch{}".format(i + 1))

    writer.close()


if __name__ == '__main__':
    train()
