"""
9.4 试编程实现k均值算法，设置三组不同的k值、三组不同初始中心点，在西瓜数据集4.0上进行实现比较，
并讨论什么样的初始中心有利于取得好结果。
"""


import pandas as pd
import numpy as np


def init_centroid(data: np.ndarray, k: int = 3):
    """
    从数据集样本中随机选择k个样本作为质心
    :param data: 数据集
    :param k: 质心个数
    :return: 质心
    """
    # 获取随机索引, replace为False表示生成的随机数不重复
    indexes = np.random.choice(np.arange(data.shape[0]), size=k, replace=False)

    # 使用随机索引
    centroids = data[indexes, :]

    return centroids


def compute_distance(x1, x2):
    """
    计算两个样本间的欧拉距离
    :param x1:
    :param x2:
    :return:
    """
    return np.sqrt(np.sum(np.square(x1 - x2)))


def k_means(data: np.ndarray, k: int, dist_meas: 'function' = compute_distance) -> (np.ndarray, np.ndarray):
    """
    k-means算法:
        1. 计算质心
        2. 分配
        3. 更新质心
    :param data: 数据集
    :param k: 质心数量
    :param dist_meas: 距离计算方法
    :return: 质心, 质心分配情况
    """
    m = data.shape[0]
    centroids = init_centroid(data, k)

    # 簇分配情况, 第一列保存当前样本所属簇的索引, 第二列保存当前样本距离簇心的距离的平方
    cluster_assignment = np.zeros((m, 2))
    cluster_changed = True

    # 当簇不再更新时结束迭代
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            # 计算样本data[i, :]与每个质心的距离并记录距离最近的质心索引
            min_distance = np.inf
            min_index = -1

            # 计算质心
            for j in range(k):
                dist_j_i = dist_meas(centroids[j, :], data[i, :])

                if dist_j_i < min_distance:
                    min_distance = dist_j_i
                    min_index = j
            if cluster_assignment[i, 0] != min_index:
                cluster_changed = True

            # 分配
            cluster_assignment[i, :] = min_index, min_distance ** 2

        # 更新质心
        for cent_index in range(k):
            points_in_cluster = data[np.nonzero(cluster_assignment[:, 0] == cent_index)[0]]
            centroids[cent_index, :] = np.mean(points_in_cluster, axis=0)\

    return centroids, cluster_assignment


def main():
    df = pd.read_csv('../../data/chapter_9/watermelon_four_point_zero.csv')
    df.drop("编号", axis=1, inplace=True)

    data = df.values.astype(float)

    centroids, cluster_assignment = k_means(data, 3, compute_distance)
    print(centroids)


if __name__ == '__main__':
    main()
