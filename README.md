### 实训——基于AI算法的智能交通标识识别项目
> 光庭实习项目

#### 1. 专业综合——MNIST手写体识别
> 本实践旨在通过基于深度学习框架`PyTorch`的手写数字识别模型，实现对`MNIST`手写数字数据集的准确识别。

要求如下：
  - 项目模块化
  - 搭建一个神经网络(神经网络不限)
  - 编写完成训练代码，推理代码，评测代码
  - 可视化推理结果和评测结果
  - 制作一个评测集，运用学习过的数据增强方法，使评测集图片数量不少于100张，使用训练好的模型进行评测，获取准确率指标
  - 至少画出`train_loss`, `val_loss`曲线，使用`tensorboard`显示出来

#### 2. 生产实习——YOLOv8交通标识识别
> 本次实验将使用`YOLOv8`训练`TT100K`中国交通标志数据集，完成一个多目标检测实战项目，可实时检测图像、视频、摄像头和流媒体(http/rtsp)中的交通标志，并提供`PySide6`开发的可视化演示界面。

`YOLOv8`识别交通标志牌项目开发要求：
  - 下载`TT100K`数据集，制作训练数据集
  - 编写`check dataset.py`脚本，对数据集进行检查，输出：
    - 每个类别的图片数量（用直方图可视化)
    - 每个类别的检测框的数量（用直方图可视化)
    - 图片总数量
    - 检测框总数量
    - 平均每张图片的检测框数量
    - 图片和标注文件是否一一对应(如有不对应的文件需要将文件路径保存下来)
    - 类别是否在指定的45类中
  - 配置参数，进行模型训练，利用`tensorboard`可视化训练`loss`和验证`loss`
  - 使用测试集图片对模型进行评估，获取评估结果
  - 利用`labelimg`标注工具，标注200张交通标志牌图片作为评测集，并使用`check_dataset.py`检查评测集
  - 编写评测代码对评测集进行评测，获取模型检测准确率
  - 使用`PySide6`创建图形用户界面进行可视化显示
  - 将训练得到的模型转化为`onnx`,并使用`onnxruntime`进行推理一张图片，获取准确结果
> VOCdevkit文件（数据集）在YOLOv8/YOLOv8_TrafficSignIdentification目录下，文件太大没有上传到GitHub   
> 阿里网盘自存档。[分享链接]( )
-----
**自存档**
