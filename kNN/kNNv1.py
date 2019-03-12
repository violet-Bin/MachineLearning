import numpy as np
from math import sqrt
from collections import Counter

"""
k: 最接近的k个数
X_train: 样本
y_train: 标签
x: 需要分析的数据
"""


def kNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of x_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to x_train"

    distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
    nearest = np.argsort(distances)  # 按索引排序，返回的结果是以索引递增的列表

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)  # 计算各元素的数量
    predict_y = votes.most_common(1)[0][0]  # 找出元素个数最多的一个元素
    print(predict_y)
