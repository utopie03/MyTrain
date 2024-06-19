import cv2
import numpy as np


# 图像腐蚀图像膨胀图像开运算图像闭运算形态学梯度
def morphology_process():
    img_ori = cv2.imread('ori.jpg')
    cv2.imshow("img_ori", img_ori)
    cv2.waitKey(0)
    # 图像腐蚀
    kernel = np.ones((5, 5))
    img_corrosion = cv2.erode(img_ori, kernel=kernel, iteratians=None)
    cv2.imshow("img_corrosion", img_corrosion)
    cv2.waitKey(0)
    # 图像膨胀
    img_dilate = cv2.dilate(img_ori, kernel=kernel, iterations=None)
    cv2.imshow("img_dilate", img_dilate)
    cv2.waitKey(0)
    # 开运算
    img_ori02 = cv2.imread(' ori02.jpg ')
    img_open = cv2.morphologyEx(img_ori02, cv2.HORPH_OPEN, kernel)
    cv2.imshow("img_open", img_open)
    cv2.waitKey(0)
    # 闭运算
    img_close = cv2.morphologyEx(img_ori02, cv2.HORPH_CLOSE, kernel)
    cv2.imshow("img_close", img_close)
    cv2.waitKey(0)

    # 形态梯度膨胀图与腐蚀图之差,能够保留物体的边缘轮廓,
    gradient = cv2.morphologyEx(img_open, cv2.HORPH_GRADIENT, kernel)
    cv2.imshow(" gradient", gradient)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vague():
    img = cv2.imread("lenna.jpg")
    # 均值模糊
    img_avg = cv2.blur(img, (11, 11))  # 这里的(11，11)是模糊的核大小
    cv2.imshow("img_ avg", img_avg)
    cv2.waitKey(0)
    # 高斯模糊
    img_gaussianblur = cv2.GaussianBlur(img, (11, 11), 0)
    cv2.imshow("img_gaussianblur", img_gaussianblur)
    cv2.waitKey(0)
    # 中值模糊
    img_median = cv2.medianBlur(img, 11)
    cv2.imshow("img_median", img_median)
    cv2.waitKey(0)
    # 双边滤波
    img_bilateral = cv2.bilateralFilter(img, 9, 100, 100)
    cv2.imshow("img_bilateral", img_bilateral)
    cv2.waitKey(0)


def contour_processing():
    img = np.zeros((256, 256, 3), np.uint8)  # 画矩形框
    cv2.rectangle(img, (120, 120), (80, 80), (0, 255, 0), thickness=2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    # 画旋转矩形辫
    img2 = np.zeros((256, 256, 3), np.uint8)
    cnt = np.array([[100, 0], [150, 50], [50, 150], [0, 100]])  # 必须是array数组的形式
    rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的(中心(x, y)，〔宽，高)，旋转角度)
    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for 0penCV 3.x)box = np.intp(box)
    # 画出来
    cv2.drawContours(img2, [box], 0, (255, 0, 0), 1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)


# if __name__ == '__main__':