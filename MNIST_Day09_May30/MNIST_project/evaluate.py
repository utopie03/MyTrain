import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from Net import *


# 加载测试集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 加载模型
model = torch.load("models/module_20.pth")
device = torch.device("cpu")

# 准备写入结果的文件
result_file = open("evaluation_results.txt", "w")

# 在测试集上评测模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    count = 0
    for images, labels in test_loader:
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
