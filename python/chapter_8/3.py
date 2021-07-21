"""
8.3 从网上下载或自己编程实现Adaboost, 以不剪枝决策树为基学习器, 在
西瓜数据集3.0上训练一个Adaboost集成, 并与图8.4进行比较。
"""


import pandas as pd
import numpy as np


class Adaboost:

    def __init__(self,
                 T: int,
                 base_learner: object):
        """
        初始化参数
        :param T: 集成器的基学习器数量
        :param base_learner: 基学习器
        """
        self.T = T
        self.base_learner = base_learner

        # 权重
        self.weights = None

        # 集成学习器
        self.ensemble_learner = []

    def fit(self, df: pd.DataFrame):
        m, n = df.shape

        # 设置初始权重
        self.weights = np.ones((self.T, n))
        self.weights[0] /= n

        # 训练T个基学习器
        for index in range(self.T):
            pass

    def predict(self, X: pd.DataFrame, y: np.ndarray = None) -> np.ndarray:
        pass


def main():
    # load data
    df = pd.read_csv('../../data/chapter_4/watermelon_dataset.csv')
    df.drop('编号', axis=1, inplace=True)
    print(df)

    ada = Adaboost(T=10)
    ada.fit(df)


if __name__ == '__main__':
    main()
