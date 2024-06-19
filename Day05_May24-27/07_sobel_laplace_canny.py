import cv2
import numpy as np

from matplotlib import pyplot as plt


def main():
    # 读取图像
    img = cv2.imread("lenna.png", 1)
    # 将图像转换为灰度图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 对灰度图像进行Sobel X方向边缘检测
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    # 对灰度图像进行Sobel Y方向边缘检测
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    # 对灰度图像进行Laplacian边缘检测
    img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
    # 对灰度图像进行Canny边缘检测
    img_canny = cv2.Canny(img_gray, 100, 150)

    # 在同一窗口中显示原始灰度图像
    plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
    # 显示Sobel X方向边缘检测结果
    plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("sobel_x")
    # 显示Sobel Y方向边缘检测结果
    plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("sobel_y")
    # 显示Laplacian边缘检测结果
    plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("laplace")
    # 显示Canny边缘检测结果
    plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")
    # 显示窗口
    plt.show()


if __name__ == "__main__":
    main()