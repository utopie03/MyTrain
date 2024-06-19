import cv2


def laplace():
    img = cv2.imread("lenna.png", 0)
    # 使用Laplacian算子对图像进行边缘检测。并将结果保存在gray_lap变量中
    # ksize参效指定了算子的大小为3
    gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    '''
    在经过处理后,别忘了用convertScaleAlbs(函数将其转回原来的uint8形式.
    否则将无法显示图像，而只是一副灰色的窗口。
    '''
    dst = cv2.convertScaleAbs(gray_lap)
    # 呈示处理后的图像
    cv2.imshow("laplacian ", dst)
    # 等待用户按键。参数为0表示无限等待
    cv2.waitKey(0)
    # 销毁所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    laplace()
