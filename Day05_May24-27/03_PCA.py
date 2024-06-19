import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris


def iris_case():
    '''主成分分析'''
    # 加载鸢尾花数据集。并返回特征和标签
    x, y = load_iris(return_X_y=True)
    # 初始化PCA对象，投置主成分数量为2
    pca = dp.PCA(n_components=2)
    # 对数据进行PCA降维处理
    reduced_x = pca.fit_transform(x)

    # 初始化不同类别的数据列表
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    # 遍历降维后的数据
    for i in range(len(reduced_x)):
        # 如果标签为0（代表鸢尾花种类为setosa)，则添加到红色数据列表
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        # 如果标签为1(代表鸢尾花种类为vensicolour），则添加到蓝色数据列表
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        # 如果标签为2（代表鸢尾花种类为virginica)，则添加到绿色数据列表
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])

    # 绘制红色数据点（setosa)
    plt.scatter(red_x, red_y, c='r', marker='x')
    # 绘制蓝色数据点（vensicolour)
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    # 绘制绿色数据点（vinginica)
    plt.scatter(green_x, green_y, c='g', marker='.')
    # 显示图表
    plt.show()


if __name__ == '__main__':
    iris_case()
