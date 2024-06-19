import numpy as np
from sklearn.decomposition import PCA

x = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9,36, 1], [2, 10, 62, 1], [3, 5, 85, 21]])
pca = PCA(n_components=2)
# 降到2维
pca.fit(x)
# 训练
newX = pca.fit_transform(x)
# 降维后的数握
# PCA(copy=True，n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  # 输出出贡献率
print(newX)  # 绘出降维后的数据
