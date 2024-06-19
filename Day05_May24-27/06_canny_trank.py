import cv2


def CannyThreshold(lowThreshold):
    # 对灰度图像进行高斯模糊处理
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    # 使用Canny算法检测边缘  双阈值算法检测
    detected_edges = cv2.Canny(detected_edges,  # 输入原图(必须为单通道图)
                                lowThreshold,  # 较小的阈值1
                                lowThreshold * ratio,  # 较大的阈值2 用于检测图像中明显的边缘
                                apertureSize=kernel_size)
    # 将检测到的边缘与原始图像进行位运算，将边缘部分保留颤色
    # cv2.bitwise_and用于对两幅二值图像进行按位"与"操作
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    # 显示处理后的图像
    cv2.imshow('canny demo', dst)


# 设置低阈值
lowThreshold = 0
# 设置高同阈值
highThreshold = 100
# 设置比率
ratio = 3
# 设置卷积核大小
kernel_size = 3
# 读取图像
img = cv2.imread('lenna.png')
# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建窗口
cv2.namedWindow('canny demo')
# 创建滑动条，用于调整低阈值
'''
cV2.createTrackbar(Track_name，inmg，min，max，TrackbarCallback)
- Track_name:滑动条的名字。
- img:滑动条所在画布。
- min:滑动条的最小值。
- max:滑动条的最大值。
- TrackbarCallback:滑动条的回调函数。"
'''
cv2.createTrackbar('Win threshold', 'canny demo', lowThreshold, highThreshold, CannyThreshold)
CannyThreshold(0)  # initialization
# 等待按下ESC提退出窗口
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()