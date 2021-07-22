"""
8.3 从网上下载或自己编程实现Adaboost, 以不剪枝决策树为基学习器, 在
西瓜数据集3.0上训练一个Adaboost集成, 并与图8.4进行比较。

代码中使用双层CART决策树作为弱分类器
数据集仅使用密度、含糖率两个连续值属性

代码参考:
https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch8--集成学习/8.3-AdaBoost.py
"""
from typing import Dict

import pandas as pd
import numpy as np


class TreeNode:
    """
    单层决策树节点
    """

    def __init__(self,
                 split_attribute: str = None,
                 split_val: float = None):
        """
        :param split_attribute: 划分属性
        :param split_val: 划分值
        """
        self.split_attribute = split_attribute
        self.split_val = split_val
        self.sub_trees = {}
        self.label = None
        self.deep = 0


def calculate_weighted_gini(y: pd.Series, weights: np.ndarray):
    """
    计算标签集y下的加权基尼值
    :param y:
    :param weights:
    :return: float
    """
    labels = pd.unique(y)
    sum_weights = np.sum(weights)

    gini = 1.0
    for label in labels:
        gini -= (np.sum(weights[y == label]) / sum_weights) ** 2

    return gini


def calculate_gini_index(attribute: str, df: pd.DataFrame, weights: np.ndarray) -> (str, float):
    attribute_values = np.sort(df.loc[:, attribute].values)
    n = attribute_values.shape[0]
    sum_weights = np.sum(weights)

    split_values = [(attribute_values[i] + attribute_values[i + 1]) / 2.0 for i in range(n - 1)]
    min_gini_index = np.inf
    best_split_val = None

    for split_value in split_values:
        y = df.iloc[:, -1]
        y_le = y[df.loc[:, attribute] <= split_value]
        y_mt = y[df.loc[:, attribute] > split_value]

        weights_le = weights[df.loc[:, attribute] <= split_value]
        weights_mt = weights[df.loc[:, attribute] > split_value]

        gini_le = calculate_weighted_gini(y_le, weights_le)
        gini_mt = calculate_weighted_gini(y_mt, weights_mt)

        gini = (np.sum(weights_le) * gini_le + np.sum(weights_mt) * gini_mt) / sum_weights

        if gini < min_gini_index:
            min_gini_index = gini
            best_split_val = split_value

    return attribute, best_split_val


def choose_split_attribute(df: pd.DataFrame, weights: np.ndarray):
    # 数据集只包含两个属性: 密度,含糖率
    attributes = df.columns[: -1]

    gini_index_0, split_val_0 = calculate_gini_index(attributes[0], df, weights)
    gini_index_1, split_val_1 = calculate_gini_index(attributes[1], df, weights)

    if gini_index_0 < gini_index_1:
        return attributes[0], split_val_0
    else:
        return attributes[1], split_val_1


def generate_decision_tree(df: pd.DataFrame, weights: np.ndarray, deep: int = 0) -> TreeNode:
    """
    生成决策树节点
    :param df: 数据，第[0, n-1]列为数据集的属性，第n列为数据标签
    :param weights: 权重
    :param deep: 树节点的深度
    :return: 决策树节点
    """
    # 当前分支下，样本数量小于等于2 或者 深度达到2时，直接设置为也节点
    if (deep == 2) | (df.shape[0] <= 2):
        tree = TreeNode()

        pos_weight = np.sum(weights[df.iloc[:, -1] == 1])
        neg_weight = np.sum(weights[df.iloc[:, -1] == -1])
        if pos_weight > neg_weight:
            tree.label = 1
        else:
            tree.label = -1

        return tree

    split_attribute, split_val = choose_split_attribute(df, weights)

    tree = TreeNode(split_attribute=split_attribute,
                    split_val=split_val)

    le = df.loc[:, split_attribute] <= split_val
    mt = df.loc[:, split_attribute] > split_val

    tree.sub_trees['le'] = generate_decision_tree(df[le], weights[le], deep + 1)
    tree.sub_trees['mt'] = generate_decision_tree(df[mt], weights[mt], deep + 1)

    return tree


def predict_base_learner(tree: TreeNode, df: pd.DataFrame) -> np.ndarray:
    """
    基于基学习器预测所有样本
    :param tree:
    :param df:
    :return:
    """
    m = df.shape[0]
    y_predict = []

    for i in range(m):
        x = df.iloc[i, :]
        y_predict.append(predict_single_sample(tree, x))

    return np.array(y_predict)


def predict_single_sample(tree: TreeNode, x: pd.Series) -> int:
    if tree.label:
        return tree.label
    else:
        if x[tree.split_attribute] <= tree.split_val:
            return predict_single_sample(tree.sub_trees['le'], x)
        else:
            return predict_single_sample(tree.sub_trees['mt'], x)


def fit_adaboost(df: pd.DataFrame, base_learner_count: int = 10):
    m, n = df.shape
    weights = np.ones(m) / m
    y = df.iloc[:, -1].values
    learner_arr = []
    alpha_arr = []
    aggregate_class_est = 0.0

    for t in range(base_learner_count):
        # 训练基学习器
        tree = generate_decision_tree(df, weights)

        # 使用基学习期预测
        predict_y = predict_base_learner(tree, df)

        # 计算错误率
        error_rate = np.sum(weights[y != predict_y])

        if error_rate > 0.5:
            break

        # 更新alpha
        alpha_t = 0.5 * np.log((1 - error_rate) / error_rate)

        learner_arr.append(tree)
        alpha_arr.append(alpha_t)

        # 更新权重
        true_indexes = np.ones(y.shape)
        true_indexes[y == predict_y] = -1

        expon = alpha_t * true_indexes
        weights = np.multiply(weights, np.exp(expon))
        weights /= np.sum(weights)

        aggregate_class_est += alpha_t * predict_y
        aggregate_y = np.sign(aggregate_class_est)
        agg_errors = np.zeros(aggregate_y.shape)
        agg_errors[aggregate_y != y] = 1
        print("iter: {}, total error: {}".format(t, agg_errors.sum() / m))

    return learner_arr, alpha_arr


def main():
    # load data
    df = pd.read_csv('../../data/chapter_4/watermelon_dataset_numeric.csv')
    df.replace(['是', '否'], [1, -1], inplace=True)
    df.iloc[:, -1].astype(int)
    df.drop('编号', axis=1, inplace=True)

    fit_adaboost(df)


if __name__ == '__main__':
    main()
