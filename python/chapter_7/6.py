"""
7.6 试编程实现AODE分类器，并以西瓜数据集3.0为训练集，对p151的"测1"样本进行判别。
"""
import sys
from typing import List

import pandas as pd
import numpy as np


class AODEClassifier(object):
    def __init__(self, m_hat: int):
        """
        初始化参数
        :param m_hat: 阈值
        """
        self.m_hat = m_hat
        # P(c, xi)
        self.p_c_xi = {}
        # P(xj| c, xi)
        self.p_xj_or_c_xi = {}

        self.labels = None
        self.labels_cnt = None
        self.attributes = None
        self.N = None
        self.supper_parent = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        训练AODE分类器
        :param X:
        :param y:
        :return:
        """
        self.labels = set(y)
        self.labels_cnt = len(self.labels)
        self.N = self.labels_cnt
        self.attributes = X.columns

        self.__init_spode(X)
        self.__compute_p_c_xi(X, y)
        self.__compute_p_xj_or_c_xi(X, y)

    def __compute_p_c_xi(self, X: pd.DataFrame, y: np.ndarray):
        """
        计算P(c, xi)
        :return:
        """
        for label in self.labels:
            self.p_c_xi[label] = {}
            for attribute in self.supper_parent:
                X_label = X[y == label]
                self.p_c_xi[label][attribute] = {}
                for attribute_value in self.supper_parent[attribute]:
                    X_label_attribute_value = X_label[X_label.loc[:, attribute] == attribute_value]
                    N_i = len(self.supper_parent[attribute])
                    self.p_c_xi[label][attribute][attribute_value] \
                        = (X_label_attribute_value.shape[0] + 1.0) / (X.shape[0] + self.N * N_i)

    def __compute_p_xj_or_c_xi(self, X: pd.DataFrame, y: np.ndarray):
        """
        计算P(xj | c,xi)
        :param X:
        :param y:
        :return:
        """
        for label in self.labels:
            self.p_xj_or_c_xi[label] = {}
            X_label = X[y == label]
            for supper_parent in self.supper_parent.keys():
                for xi in self.supper_parent[supper_parent]:
                    self.p_xj_or_c_xi[label][xi] = {}
                    X_label_xi = X_label[X_label.loc[:, supper_parent] == xi]

                    for attribute in self.attributes:
                        attribute_values = pd.unique(X.loc[:, attribute])
                        N_j = attribute_values.shape[0]
                        for xj in attribute_values:
                            X_label_xi_xj = X_label_xi[X_label_xi.loc[:, attribute] == xj]
                            self.p_xj_or_c_xi[label][xi][xj] \
                                = (X_label_xi_xj.shape[0] + 1.0) / (X_label_xi.shape[0] + N_j)

    def __init_spode(self, X: pd.DataFrame):
        """
        初始化超父
        :param X:
        :return:
        """
        for attribute in self.attributes:
            self.supper_parent[attribute] = []
            attribute_values = pd.unique(X.loc[:, attribute])
            for attribute_value in attribute_values:
                # 获取当数据的属性attribute取值为attribute_value时数据的长度
                m = X[X.loc[:, attribute] == attribute_value].shape[0]
                if m >= self.m_hat:
                    self.supper_parent[attribute].append(attribute_value)

    def predict(self, X: pd.DataFrame) -> List[str]:
        """
        预测X中样本的类别
        取对数计算避免下溢
        :param X:
        :return:
        """
        y_predict = []
        max_p = -sys.maxsize
        best_label = None
        m = X.shape[0]

        for index in range(m):
            x_ = X.iloc[index, :]
            for label in self.labels:
                p_c_or_x = 0.0
                for supper_parent in self.supper_parent:
                    for xi in self.supper_parent[supper_parent]:
                        sub_p = np.log(self.p_c_xi[label][supper_parent][xi])
                        for xj_attribute in self.attributes:
                            xj = x_[xj_attribute]
                            sub_p += np.log(self.p_xj_or_c_xi[label][xi][xj])
                        p_c_or_x += sub_p
                if p_c_or_x > max_p:
                    max_p = p_c_or_x
                    best_label = label
            y_predict.append(best_label)

        return y_predict


def main():
    df = pd.read_csv('../../data/chapter_4/watermelon_dataset.csv')
    y = df.iloc[:, -1].values
    # todo 对连续值进行分箱将其离散化
    X = df.drop(['编号', '密度', '含糖率', '好瓜'], axis=1)

    model = AODEClassifier(0)
    model.fit(X, y)

    df_test = pd.read_csv('../../data/chapter_7/test1.csv')
    df_test.drop(['编号', '密度', '含糖率', '好瓜'], axis=1, inplace=True)
    y_predict = model.predict(df_test)
    print("预测结果为: {}。".format(y_predict[0]))


if __name__ == '__main__':
    main()
