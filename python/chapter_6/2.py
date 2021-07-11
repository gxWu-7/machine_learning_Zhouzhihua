"""
6.2 试使用LIBSVM，在西瓜数据集3.0a上分别用线性核和高斯核训练一个SVM，并比较其支持向量的差别。
LIBSVM: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
data: ../../data/chapter_4/watermelon_dataset_numeric.csv
"""


import pandas as pd
from libsvm.svmutil import *


if __name__ == '__main__':
    # load data
    path = '../../data/chapter_4/watermelon_dataset_numeric.csv'
    df = pd.read_csv(path)
    X = df.iloc[:, [1, 2]].values
    y = df.iloc[:, 3].values
    y[y == '是'] = 1
    y[y == '否'] = -1

    # train
    """
    输出参数说明:
    iter: 迭代次数
    nu: 选择的核函数类型的参数
    obj: 为SVM文件转换为的二次规划求解得到的最小值
    rho: 为判决函数的偏置项b
    nSV: 为标准支持向量个数(0<a[i]<c)
    nBSV为边界上的支持向量个数(a[i]=c)
    Total nSV为支持向量总个数
    """
    # 线性核
    svm_linear_kernel = svm_train(y, X, '-t 0')

    # 高斯核
    svm_gauss_kernel = svm_train(y, X, '-t 2')
