import cv2
import numpy as np


def bilinear_interpolation(img, out_dim):
    # 获取源图像的高度、宽度和通道数
    src_h, src_w, channel = img.shape
    # 获取目标图像的高度和宽度
    dst_h, dst_w = out_dim[1], out_dim[0]
    # 打印逐图像和目标图像的尺寸
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # 如果源图像和目标图像尺寸相同，则直接返回源图像的副本
    if src_h == dst_h and src_w == dst_w:
        return img.copyO
    # 创建目标图像，并初始化为0
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    # 计算源图像和目标图像在x和ly方向上的增放比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    # 逾历每个通道
    for i in range(3):
        # 逾历目标图像的每个像素点
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way,src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找到用于计算插值的点的坐标
                # find the coordinates of the points which will be used to compute the interpolation)
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算插值结果
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imread('bilinear interp', dst)
    cv2.waitKey(0)