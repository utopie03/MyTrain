import torchvision
from PIL import Image
from LeNet import *


def predict():
    image_path = "./CIFAR10/cifar10/test/0_10.jpg"
    image = Image.open(image_path)
    print(image)
    # 预处理
    image = image.convert('RGB')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
    ])
    image = transform(image)
    print("image shape: ", image.shape)

    model = torch.load("LeNet/module_20.pth", map_location=torch.device('cpu'))
    print("model: ", model)
    image = torch.reshape(image, (1, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(image)
    print("output: ", output)

    print(output.argmax(1))


if __name__ == '__main__':
    predict()
