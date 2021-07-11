"""
6.8 以西瓜数据集3.0的'密度'为输入，'含糖量'为输出，试使用LIBSVM训练一个SVR。
"""

import pandas as pd
from libsvm.svmutil import *


if __name__ == '__main__':
    df = pd.read_csv('../../data/chapter_4/watermelon_dataset_numeric.csv')
    X = df.iloc[:, 1].values.reshape(-1, 1)
    y = df.iloc[:, 2].values

    svm = svm_train(y, X, '-s 4')
