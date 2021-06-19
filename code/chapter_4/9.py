"""
试将4.4.2节对缺失值的处理机制推广到基尼指数计算中去。

这里仅实现基于未剪枝的方式
"""


from typing import Dict

import pandas as pd
import numpy as np


class GiniIndexDecisionTree(object):
    def __init__(self, is_leaf: bool, sub_trees: str or Dict, attribute: str = None) -> None:
        """
        决策树节点
        :param is_leaf: 是否是叶子节点
        :param sub_trees: 节点的子树，如果当前节点是叶子节点，那么子树为标签（例如：好/坏，是/否），
                            如果当前节点是非叶子节点，那么子树为{attribute_取值x： GiniIndexDecisionTree节点x}
        :param attribute: 节点划分的属性，如果当前节点不需要划分子节点，那么为None
        """
        self.is_leaf = is_leaf
        self.sub_trees = sub_trees
        self.attribute = attribute

    @staticmethod
    def compute_gini_index(df: pd.DataFrame, attribute: str) -> float:
        """
        计算df按照attribute划分后的基尼指数
        :param df:
        :param attribute: str
        :return: 基尼指数(Gini_index)
        """
        attribute_values = set(df.loc[:, attribute])
        gini_index = 0.0
        for attribute_value in attribute_values:
            sub_df = df[df.loc[:, attribute] == attribute_value]
            sub_df_gini = GiniIndexDecisionTree.compute_gini(sub_df)
            sub_df_p = sub_df.shape[0] * 1.0 / df.shape[0]
            gini_index += sub_df_p * sub_df_gini
        return gini_index

    @staticmethod
    def compute_gini(df: pd.DataFrame) -> float:
        """
        计算数据集的基尼
        :param df:
        :return: 基尼(Gini)
        """
        labels = set(df.iloc[:, -1])
        gini_df = 1.0
        for label_k in labels:
            p_k = df[df.iloc[:, -1] == label_k].shape[0] * 1.0 / df.shape[0]
            gini_df -= p_k * p_k

        return gini_df

    @staticmethod
    def generate_decision_tree_without_pruning(df: pd.DataFrame):
        """
        递归生成决策树并使用预剪枝技术，主体逻辑与不剪枝类似，不同的是在计算选择完最佳的分裂属性后，使用验证集来测试
        不划分的决策树与划分后的决策树的精度，如果划分子树后的精度小于不划分的精度，那么选择不划分，正常生成子树。
        :param df:
        :return: 决策树节点
        """
        # 获取数据集的标签类别
        labels = set(df.iloc[:, -1])

        # 获取数据集的属性名
        attributes = df.columns[: -1]

        # 如果数据集标签类别取值唯一
        if len(labels) == 1:
            return GiniIndexDecisionTree(is_leaf=True, sub_trees=df.iloc[0, -1])

        # 如果数据集属性集为空或者仅有一个属性并且取值唯一
        if len(attributes) == 0 or (len(attributes) == 1 and len(set(df.loc[:, 0])) == 1):
            max_count_label = pd.value_counts(df.iloc[:, -1]).index[0]
            return GiniIndexDecisionTree(is_leaf=True, sub_trees=max_count_label)

        best_split_attribute = None
        min_gini_index = np.inf
        # 挑选基尼指数最小的属性
        for attribute in attributes:
            cur_gini_index = GiniIndexDecisionTree.compute_gini_index(df, attribute)

            if cur_gini_index < min_gini_index:
                min_gini_index = cur_gini_index
                best_split_attribute = attribute
            # print('{}: {}'.format(attribute, cur_gini_index))
        # print('\n')

        # 使用最佳属性进行节点划分
        attribute_values = set(df.loc[:, best_split_attribute])
        tree = GiniIndexDecisionTree(is_leaf=False, sub_trees={}, attribute=best_split_attribute)
        for attribute_value in attribute_values:
            sub_df = df[df.loc[:, best_split_attribute] == attribute_value].copy()
            sub_df.drop(best_split_attribute, axis=1, inplace=True)
            tree.sub_trees[attribute_value] = GiniIndexDecisionTree.generate_decision_tree_without_pruning(sub_df)

        return tree

    def predict(self, df: pd.DataFrame) -> float:
        """
        这里的predict仅返回对df进行预测后与正确的类别对比的精度
        :param self:
        :param df:
        :return: 对df进行预测后的精度
        """
        true_count = 0.0
        for index in range(df.shape[0]):
            x = df.iloc[index, :]
            y = df.iloc[index, -1]
            predict_y = self._predict(x)
            true_count += 1 if y == predict_y else 0
        return true_count / df.shape[0]

    def _predict(self, x: pd.Series):
        """
        根据决策树判断样本属于哪一个类别
        :param x: df中的一个样本
        :return:
        """
        if self.is_leaf:
            # 如果节点为叶子节点，返回标签
            return self.sub_trees
        else:
            # 根据当前节点属性的取值进入子树递归调用_predict
            return self.sub_trees[x[self.attribute]]._predict(x)


def main():
    # 主函数
    # 读取训练集数据和验证集数据
    df_train = pd.read_csv('../../data/chapter_4/watermelon_train.csv')
    df_validate = pd.read_csv('../../data/chapter_4/watermelon_validate.csv')

    # 丢弃 '编号' 这一列
    df_train.drop('编号', axis=1, inplace=True)
    df_validate.drop('编号', axis=1, inplace=True)

    # 使用西瓜训练集生成未剪枝的决策树
    tree_without_pruning = GiniIndexDecisionTree.generate_decision_tree_without_pruning(df_train)
    acc_without_pruning = tree_without_pruning.predict(df_validate)
    print("acc_without_pruning: ", acc_without_pruning)


if __name__ == '__main__':
    main()
