"""
4.3 试编程实现基于信息熵进行划分选择的决策树算法，并为表4.3中数据生成一颗决策树
"""


import pandas as pd
import numpy as np


class ID3DecisionTree(object):
    def __init__(self, is_leaf, sub_trees, attribute=None):
        """
        决策树节点
        :param is_leaf: 节点是否为树的叶子节点
        :param attribute: 当前节点的子树根据什么属性来进行划分
        :param sub_trees: 节点的子树
        """
        self.is_leaf = is_leaf
        self.attribute = attribute
        self.sub_trees = sub_trees

    @staticmethod
    def compute_information_entropy(df: pd.DataFrame):
        """
        # todo 处理连续值
        formula 4.1
        :param df:
        :return:
        """
        labels = set(df.iloc[:, -1])
        information_entropy = 0.0
        for label_k in labels:
            p_k = len(df[df.iloc[:, -1] == label_k]) * 1.0 / df.shape[0]
            information_entropy -= p_k * np.log2(p_k) if p_k != 0 else 0
        return information_entropy

    @staticmethod
    def generate_decision_tree(df: pd.DataFrame):
        """
        根据传入的数据递归生成决策树
        :param df: 传入的数据为pd.DataFrame对象，前n列为数据的属性，最后一列为数据的标签类别
                [attr_1, attr_2, ..., attr_n, y]
        :return: ID3DecisionTree对象
        """
        # 获取数据的属性以及类别
        attributes = df.columns[: -1]
        labels = set(df.iloc[:, -1])

        # 如果数据样本全部属于同一个类别，该节点不用划分，设置为叶子节点后返回
        if len(labels) == 1:
            return ID3DecisionTree(is_leaf=True,
                                   sub_trees=df.iloc[0, 0])

        # 如果属性为空集或者仅有一个属性且取值相同，返回样本数最多的类
        if len(attributes) == 0 or (len(attributes == 1) and len(set(df.iloc[:, 0])) == 1):
            max_count_label = pd.value_counts(df.iloc[:, -1]).index[0]
            return ID3DecisionTree(is_leaf=True,
                                   sub_trees=max_count_label)

        # 从attributes中选择最佳的划分属性
        best_split_attribute = None
        cur_data_information_entropy = ID3DecisionTree.compute_information_entropy(df)
        max_entropy_gain = -np.inf
        for attribute in attributes:
            attribute_values = set(df.loc[:, attribute])
            split_information_entropy = 0.0
            for attribute_value in attribute_values:
                sub_df_k = df[df.loc[:, attribute] == attribute_value]
                p_k = sub_df_k.shape[0] * 1.0 / df.shape[0]
                split_information_entropy += p_k * ID3DecisionTree.compute_information_entropy(sub_df_k)

            cur_entropy_gain = cur_data_information_entropy - split_information_entropy
            if cur_entropy_gain > max_entropy_gain:
                best_split_attribute = attribute
                max_entropy_gain = cur_entropy_gain

        # 开始划分树节点
        tree = ID3DecisionTree(is_leaf=False,
                               attribute=best_split_attribute,
                               sub_trees={})
        best_split_attribute_values = set(df.loc[: best_split_attribute])
        for attribute_value in best_split_attribute_values:
            # 挑选出原数据集中在属性best_split_attribute取值为attribute_value的子df
            sub_df = df[df.loc[:, best_split_attribute] == attribute_value].copy()

            # 选择的子df需要去掉best_split_attribute这一列
            sub_df.drop(best_split_attribute, axis=1, inplace=True)

            # 递归生成子树
            tree.sub_trees[attribute_value] = ID3DecisionTree.generate_decision_tree(sub_df)
