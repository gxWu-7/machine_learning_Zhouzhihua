"""
8.5 试编程实现Bagging，以决策树桩为基学习器，在西瓜数据集3.0a上训练一个Bagging集成，并与图8.6进行比较。

这里的单层决策树生成函数参照了《机器学习实战》中的代码
"""


import numpy as np
import pandas as pd
from sklearn.utils import resample


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq) -> np.ndarray:
    """
    使用决策树桩根据阈值进行分类.
    :param data_matrix: 数据矩阵
    :param dimen: 数据的第几列
    :param thresh_val: 分类阈值
    :param thresh_ineq: 这里使用thresh_ineq来判断是使用小于等于(lt)还是大于(gt)，如果传入lt，
                        那么小于等于阈值的结果数组置为-1，反之将大于阈值的索引置为-1。
    :return: 分类的结果
    """
    ret_array = np.ones((np.shape(data_matrix)[0], 1))

    # 分类
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0

    # 返回结果
    return ret_array


def build_stump(data_matrix, class_labels):
    """
    建立决策树桩
    :param data_matrix: 数据矩阵
    :param class_labels: 数据集的标签
    :return: 决策树桩
    """
    label_matrix = np.mat(class_labels).T
    m, n = np.shape(data_matrix)

    best_stump = {}
    min_error = np.inf
    num_steps = 10.0

    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps

        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_matrix] = 0
                err = np.mean(err_arr)
                print("err: ", err)

                if err < min_error:
                    min_error = err
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal

    return best_stump


def bagging_train_decision_stump(data_matrix: np.ndarray, class_labels: np.ndarray, num_it: int = 40):
    weak_class_arr = []
    random_seed = 16

    for _ in range(num_it):
        # 使用sklearn的自助采样函数
        X, y = resample(data_matrix, class_labels, random_state=random_seed)
        random_seed += 1

        # 建立决策树桩
        weak_class_arr.append(build_stump(X, y))

    return weak_class_arr


def bagging_predict_decision_stump(model, X, y):
    result = np.zeros((y.shape[0], 1))

    for weak_classifier in model:
        cur_res = stump_classify(X,
                                 weak_classifier['dim'],
                                 weak_classifier['thresh'],
                                 weak_classifier['ineq'])

        result += cur_res

    return np.sign(result)


def main():
    df = pd.read_csv('../../data/chapter_4/watermelon_dataset_numeric.csv')
    df.drop('编号', axis=1, inplace=True)

    X = df.iloc[:, :2].values
    y = df.iloc[:, -1].replace(['是', '否'], [1, -1]).values

    model = bagging_train_decision_stump(X, y)

    predict_res = bagging_predict_decision_stump(model, X, y)

    print(predict_res)


if __name__ == '__main__':
    main()
