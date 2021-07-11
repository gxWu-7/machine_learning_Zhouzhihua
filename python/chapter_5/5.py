"""
5.5 试编程实现标准BP算法和累计BP算法，在西瓜数据集3.0上分别用这两个算法训练一个单隐层网络，并进行比较。

西瓜数据集放在../../data/chapter_4/watermelon_dataset.csv
"""

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_square_loss(y_hat: np.ndarray, y: np.ndarray):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return 0.5 * np.square(y_hat - y)


class Net:
    def __init__(self,
                 input_layer_neurons_count: int,
                 hidden_layer_neurons_count: int,
                 output_layer_neurons_count: int,
                 learning_rate: float) -> None:
        """
        初始化单隐层神经网络的参数
        :param input_layer_neurons_count: 输入层神经元的个数
        :param hidden_layer_neurons_count: 隐层神经元的个数
        :param output_layer_neurons_count: 输出层神经元的个数
        """
        # attribute
        # 输入层第i个神经元与隐层第h个神经元之间的连接权为v[i, h]
        self.v = np.random.random((input_layer_neurons_count, hidden_layer_neurons_count))

        # 隐层神经元的输入
        self.alpha = None

        # 隐层神经元的阈值
        self.gamma = np.random.random((1, hidden_layer_neurons_count))

        # 隐层神经元的输出
        self.b = None

        # 隐层神经元与输出层神经元的权重
        self.w = np.random.random((hidden_layer_neurons_count, output_layer_neurons_count))

        # 输出层神经元的输入
        self.beta = None

        # 输出层神经元的阈值
        self.theta = np.random.random((1, output_layer_neurons_count))

        # 输出层神经元的输出
        self.y_hat = None

        # 输出层梯度项g
        self.g = None

        # 隐层神经元梯度项e
        self.e = None

        self.learning_rate = learning_rate

    def fit_standard_bp(self, X: np.ndarray, y_: np.ndarray, max_iter: int = 20) -> None:
        """
        用标准BP算法训练
        :param X: 训练集(不包含标签)
        :param y_: 标签
        :param max_iter: 最大训练的回合数
        :return: None
        """
        for iter_ in range(max_iter):
            E = 0.0
            for index in range(X.shape[0]):
                # 计算y_hat
                x = X[index]
                y = y_[index]
                self.__input_layer_forward_compute(x)
                self.__hidden_layer_forward_compute()
                y_hat = self.__output_layer_forward_compute()

                # 计算误差
                E += compute_square_loss(y_hat, y)

                # 计算输出层梯度项
                self.__output_layer_bp_compute(y_hat=float(y_hat), y=y)

                # 计算隐层梯度项
                self.__hidden_layer_bp_compute()

                # 更新连接权和阈值
                self.__update_parameters(x)
            print('iter: {}, e: {}.'.format(iter_, E / X.shape[0]))

    def fit_accumulated_bp(self, X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> None:
        """
        用累积训练法训练，训练完所有的数据再更新参数
        :param X: 训练集(不包含标签)
        :param y: 标签
        :param max_iter: 最大训练的回合数
        :return: None
        """
        for iter_ in range(max_iter):
            self.__input_layer_forward_compute(X)
            self.__hidden_layer_forward_compute()
            y_hat = self.__output_layer_forward_compute()

            # 计算误差
            E = compute_square_loss(y_hat, y).sum(axis=0) / X.shape[0]

            # 计算输出层梯度项
            self.__output_layer_bp_compute(y_hat=y_hat, y=y)

            # 计算隐层梯度项
            self.__hidden_layer_bp_compute()

            # 更新连接权和阈值
            x = X.sum(axis=0) / X.shape[0]
            self.__update_parameters(x)

            print('iter: {}, e: {}.'.format(iter_, E / X.shape[0]))

    def __input_layer_forward_compute(self, X: np.ndarray) -> None:
        """
        计算隐层神经元的输入alpha
        :param X: X.shape = (1, input_layer_neurons_count)
        :return: None
        """
        self.alpha = np.dot(X, self.v)

    def __hidden_layer_forward_compute(self) -> None:
        """
        使用输入层神经元的输出并计算输出层的输入
        :return: None
        """
        # alpha.shape = (1, hidden_layer_neurons_count)
        self.b = sigmoid(self.alpha - self.gamma)

    def __output_layer_forward_compute(self) -> np.ndarray:
        """
        使用隐层神经元的输出并计算神经网络最终的输出
        :return: y_hat
        """
        # b.shape = (1, hidden_layer_neurons_count)
        # w.shape = (hidden_layer_neurons_count, output_layer_neurons_count)
        self.beta = np.dot(self.b, self.w)

        y_hat = sigmoid(self.beta - self.theta)
        y_hat = y_hat.reshape(-1, 1)
        return y_hat

    def __output_layer_bp_compute(self, y_hat, y) -> None:
        """
        根据式5.10计算输出层神经元梯度项g
        :return:
        """
        y = y.reshape(-1, 1)
        self.g = y_hat * (1 - y_hat) * (y - y_hat)
        self.g = self.g.sum(axis=0) / self.g.shape[0]

    def __hidden_layer_bp_compute(self) -> None:
        """
        根据式5.15计算隐层神经元梯度项e
        :return:
        """
        self.e = self.b * (1 - self.b) * (np.dot(self.g, self.w.transpose()))
        self.e = self.e.sum(axis=0) / self.e.shape[0]

    def __update_parameters(self, x: np.ndarray):
        # 根据5.11更新输出层和隐层神经元的连接权w
        # w.shape = (hidden_layer_neurons_count, output_layer_neurons_count)
        d_w = self.learning_rate * self.g * self.b
        d_w = d_w.sum(axis=0) / d_w.shape[0]

        # 根据5.12更新输出层神经元的阈值
        d_theta = -self.learning_rate * self.g

        # 根据5.13更新隐层和输入层的神经元的连接权v
        # v.shape = (input_layer_neurons_count, hidden_layer_neurons_count)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.e.ndim == 1:
            self.e = self.e.reshape(1, -1)
        d_v = self.learning_rate * np.dot(x.transpose(), self.e)

        # 根据5.14更新隐层神经元的阈值
        d_gamma = -self.learning_rate * self.e

        # update
        d_w = d_w.reshape(self.w.shape[0], self.w.shape[1])
        self.w += d_w
        self.theta += d_theta
        self.v += d_v
        self.gamma += d_gamma


def main():
    # 导入西瓜3.0数据集
    df = pd.read_csv('../../data/chapter_4/watermelon_dataset.csv')

    # 处理数据
    # 丢弃编号列
    df.drop('编号', axis=1, inplace=True)
    # one-hot编码处理类别信息
    data = pd.get_dummies(df, columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    data['好瓜'].replace(['是', '否'], [1, 0], inplace=True)

    X = data.drop('好瓜', axis=1).astype(float).values
    y = data.loc[:, '好瓜'].values

    np.random.seed(1)
    # 标准BP
    model1 = Net(input_layer_neurons_count=X.shape[1],
                 hidden_layer_neurons_count=20,
                 output_layer_neurons_count=1,
                 learning_rate=0.15)
    model1.fit_standard_bp(X, y, max_iter=200)

    # 累积BP
    model2 = Net(input_layer_neurons_count=X.shape[1],
                 hidden_layer_neurons_count=20,
                 output_layer_neurons_count=1,
                 learning_rate=0.15)
    model2.fit_accumulated_bp(X, y)


if __name__ == '__main__':
    main()
