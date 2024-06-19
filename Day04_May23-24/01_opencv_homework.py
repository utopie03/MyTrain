import cv2

# VideoCapture方法是cv2库提供的视频方法
cap = cv2.VideoCapture('1.mp4')
# 设置雷要保存视频的格式"xvid”
# 该参数是MPEG-4编码类型，文件名后缀为.avi mp4格式: *'mp4v'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 设置视频帧频
fps = cap.get(cv2.CAP_PROP_FPS)
# 设置视频大小
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# videoWriter方法是cv2库提供的保存视频方法
# 按照设置的格式来out输出
out = cv2.VideoWriter("output1.avi", fourcc, fps, size)

# 确定视频打开并循环读取
while (cap.isOpened()):
    # 逐帧读取，ret返国布尔值
    # 参数ret为True或者False,代表有没有读取到图片
    # frame表示截取到一帧的图片
    ret, frame = cap.read()
    if ret == True:
        cv2.putText(frame,  # 图像
                    text="csq",  # 文字
                    org=(508, 380),  # 位置
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                    color=(0, 0, 255),
                    fontScale=1,  # 字体大小
                    thickness=2,  # 字体相细
                    )
        out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            break
# 释放资源
cap.release()
out.release()
# 关闭窗口
cv2.destroyAllWindows()
