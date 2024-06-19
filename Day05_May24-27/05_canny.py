import cv2


def main():
    # 读取图像
    img = cv2.imread("lenna.png", 1)
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用Canny算法进行边缘检测
    canny = cv2.Canny(gray, 200, 300)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()