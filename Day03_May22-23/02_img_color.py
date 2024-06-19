import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.jpg', 0)

# 1.访问图像属性
print(img.shape)

# 2.彩色图像转化为灰度图像
img1 = cv2.imread('lenna.jpg')
cv2.cvtColor((img1, cv2.COLOR_BGR2RGB))

# 3.色彩通道操作
b, g, r = cv2.split(img1)
print("b通道：", b)
print("g通道：", g)
print("r通道：", r)
