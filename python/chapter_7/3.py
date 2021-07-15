"""
7.3 试编程实现拉普拉斯修正的朴素贝叶斯分类器，并以西瓜数据集3.0为训练集，对p.151"测1"样本进行判别。

西瓜数据集3.0: ../../data/chapter_4/watermelon_dataset.csv
"""
from typing import List

import pandas as pd
import numpy as np
import sys


def is_continuous(df: pd.DataFrame, attribute: str) -> bool:
    """
    判断数据属性对应的取值是否是连续值。由于西瓜数据集中数值只有np.float，故只需判断数据类型
    是不是np.float，如果是返回True，否则返回False。
    :param df:
    :param attribute: 数据的属性列名
    :return:
    """
    if df.loc[:, attribute].shape[0] == 0:
        raise ValueError("df.shape[0] == 0.")

    if type(df.loc[:, attribute][0]) == np.float:
        return True
    else:
        return False


class BayesClassifier:

    def __init__(self):
        # 保存概率
        self.p = {}
        self.p_label = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        训练朴素贝叶斯模型
        对数据中的每一种类别:
            对数据中的每一种属性:
                对数据中的属性的每一种取值:
                    估算概率(使用拉普拉斯修正)
                    将概率保存在self.p中

        :param df: 最后一列为样本的类别，其它均为样本属性
        :return:
        """
        labels = pd.unique(df.iloc[:, -1])
        attributes = df.columns[: -1]

        # 遍历所有类
        for label in labels:
            self.p[label] = {}

            # 在df中筛选标签为label的数据
            df_label = df[df.iloc[:, -1] == label]
            self.p_label[label] = (df_label.shape[0] + 1.0) / (df.shape[0] + len(labels))

            # 遍历所有属性
            for attribute in attributes:
                self.p[label][attribute] = {}

                if is_continuous(df, attribute):
                    # 如果该属性取值为连续值
                    # 计算均值和方差
                    self.p[label][attribute]['mean'] = np.mean(df_label.loc[:, attribute].values)
                    self.p[label][attribute]['var'] = np.var(df.label.loc[:, attribute].values)

                else:
                    # 如果该属性取值为离散值
                    attribute_values = pd.unique(df.loc[:, attribute])

                    # 遍历某个属性对应的所有取值
                    for attribute_value in attribute_values:
                        # 在df_label中筛选属性attribute取值为attribute_value的数据
                        df_attribute = df_label[df_label.loc[:, attribute] == attribute_value]

                        # 计算概率
                        self.p[label][attribute][attribute_value] \
                            = (df_attribute.shape[0] + 1.0) / (df_label.shape[0] + len(attribute_values))

    def predict(self, x: pd.DataFrame) -> List[str]:
        """
        预测x的类别
        :param x: 只包含属性，不包含标签的数据样本
        :return: 预测值的列表
        """
        y_pred = []
        m = x.shape[0]
        # 概率取对数计算，避免下溢
        attributes = x.columns
        for index in range(m):
            max_p = -sys.maxsize
            best_label = None
            for label in self.p_label.keys():
                p = np.log(self.p_label[label])
                for attribute in attributes:
                    x_ = x.iloc[index, :]

                    if is_continuous(x, attribute):
                        """
                        将公式7.18分为左右两个部分计算，记:
                        p_left = 1.0 / sqrt(2 * pi) * var_c_i
                        p_right = exp(-square(x_i - mean_c_i) / (2 * square(var_c_i)))
                        """
                        p_left = 1.0 / np.sqrt(2 * np.pi) * self.p[label][attribute]['var']
                        p_right = np.exp(-np.square(x_[attribute] - self.p[label][attribute]['mean'] / \
                                                    (2 * np.square(self.p[label][attribute]['var']))))
                        p += np.log(p_left * p_right)
                    else:
                        p += np.log(self.p[label][attribute][x_[attribute]])

                if p > max_p:
                    max_p = p
                    best_label = label
            y_pred.append(best_label)
        return y_pred


def main():
    # load data
    df = pd.read_csv('../../data/chapter_4/watermelon_dataset.csv')
    df.drop('编号', inplace=True, axis=1)

    # 训练朴素贝叶斯分类器
    model = BayesClassifier()
    model.fit(df)

    # 训练出来的概率与书上一致
    print(model.p)
    print(model.p_label)

    # 测试
    df_test = pd.read_csv('../../data/chapter_7/test1.csv')
    df_test.drop(['编号', '好瓜'], axis=1, inplace=True)
    y_pred = model.predict(df_test)
    test_label = '好瓜' if y_pred[0] == '是' else '坏瓜'
    print("预测值 = {}, 因此该瓜是{}".format(y_pred[0], test_label))


if __name__ == '__main__':
    main()
