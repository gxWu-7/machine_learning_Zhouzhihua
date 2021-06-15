"""
4.3 试编程实现基于信息熵进行划分选择的决策树算法，并为表4.3中数据生成一颗决策树

注: 表4.3中的数据对应于data/chapter_4/watermelon_dataset.csv
"""

import pandas as pd
import numpy as np


class ID3DecisionTree(object):
    def __init__(self,
                 is_leaf,
                 sub_trees,
                 attribute=None,
                 is_continuous=False,
                 continuous_split_value=0):
        """
        决策树节点
        :param is_leaf: 节点是否为树的叶子节点
        :param attribute: 当前节点的子树根据什么属性来进行划分
        :param sub_trees: 节点的子树
        :param is_continuous: 当前节点的属性是否是连续值
        :param continuous_split_value: 连续值的划分值
        """
        self.is_leaf = is_leaf
        self.attribute = attribute
        self.sub_trees = sub_trees
        self.is_continuous = is_continuous
        self.continuous_split_value = continuous_split_value

    @staticmethod
    def compute_information_entropy(df: pd.DataFrame):
        """
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
    def get_discontinuous_entropy_gain(df: pd.DataFrame, attribute: str):
        """
        处理非连续属性，计算按非连续值属性划分后的信息增益
        :param df:
        :param attribute:
        :return:
        """
        # 获取数据集在属性为attribute时的取值(集合)
        attribute_values = set(df.loc[:, attribute])
        split_information_entropy = 0.0
        cur_data_information_entropy = ID3DecisionTree.compute_information_entropy(df)

        for attribute_value in attribute_values:
            sub_df_k = df[df.loc[:, attribute] == attribute_value]
            p_k = sub_df_k.shape[0] * 1.0 / df.shape[0]
            split_information_entropy += p_k * ID3DecisionTree.compute_information_entropy(sub_df_k)

        cur_entropy_gain = cur_data_information_entropy - split_information_entropy
        return cur_entropy_gain

    @staticmethod
    def get_continuous_entropy_gain_and_attribute_split_value(df: pd.DataFrame, attribute: str):
        """
        处理连续值属性，计算按连续值属性划分后的信息增益以及最佳划分值
        :param df:
        :param attribute:
        :return:
        """
        # 获取数据集attribute列的数据
        continuous_values = list(df.loc[:, attribute])

        # 从小到大排序
        continuous_values.sort()

        # 选取最优的划分点
        dataset_information_entropy = ID3DecisionTree.compute_information_entropy(df)
        best_split_value = 0
        max_entropy_gain = -np.inf
        for index in range(0, len(continuous_values) - 1):
            split_value = (continuous_values[index] + continuous_values[index + 1]) / 2.0

            df_less = df[df.loc[:, attribute] < split_value]
            df_more_or_equal = df[df.loc[:, attribute] >= split_value]

            entropy_less = ID3DecisionTree.compute_information_entropy(df_less)
            entropy_more_or_equal = ID3DecisionTree.compute_information_entropy(df_more_or_equal)

            p_less = df_less.shape[0] * 1.0 / df.shape[0]
            p_more_or_equal = df_more_or_equal.shape[0] * 1.0 / df.shape[0]

            split_information_entropy = p_less * entropy_less + p_more_or_equal * entropy_more_or_equal
            split_entropy_gain = dataset_information_entropy - split_information_entropy

            if split_entropy_gain > max_entropy_gain:
                max_entropy_gain = split_entropy_gain
                best_split_value = split_value
        return max_entropy_gain, best_split_value

    @staticmethod
    def generate_decision_tree(df: pd.DataFrame, has_continuous_attribute: bool = True):
        """

        :param df: 传入的数据为pd.DataFrame对象，前n列为数据的属性，最后一列为数据的标签类别
                [attr_1, attr_2, ..., attr_n, y]
        :param has_continuous_attribute: 是否包含连续值属性
        :return: ID3DecisionTree对象
        """
        if has_continuous_attribute:
            return ID3DecisionTree.generate_decision_tree_with_continuous_attribute(df)
        else:
            return ID3DecisionTree.generate_decision_tree_without_continuous_attribute(df)

    @staticmethod
    def generate_decision_tree_with_continuous_attribute(df: pd.DataFrame):
        """
        递归生成决策树，加入书中4.4.1连续值处理部分内容
        函数逻辑与generate_decision_tree_without_continuous_attribute类似
        :param df:
        :return:
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
        max_entropy_gain = -np.inf
        attribute_is_continuous = False
        best_split_value = 0
        for attribute in attributes:
            # 判断该属性是否是连续值
            if df.loc[:, attribute].dtype == np.float or df.loc[:, attribute].dtype == np.int:
                # 如果该属性取值为连续值
                cur_entropy_gain, split_value \
                    = ID3DecisionTree.get_continuous_entropy_gain_and_attribute_split_value(df, attribute)

                if cur_entropy_gain > max_entropy_gain:
                    best_split_attribute = attribute
                    max_entropy_gain = cur_entropy_gain
                    attribute_is_continuous = True
                    best_split_value = split_value

            else:
                # 如果该属性取值为非连续值
                cur_entropy_gain = ID3DecisionTree.get_discontinuous_entropy_gain(df, attribute)

                if cur_entropy_gain > max_entropy_gain:
                    best_split_attribute = attribute
                    max_entropy_gain = cur_entropy_gain
                    attribute_is_continuous = False
                    best_split_value = np.nan
            print('{}: {}'.format(attribute, cur_entropy_gain))

        # 开始划分树节点
        tree = ID3DecisionTree(is_leaf=False,
                               attribute=best_split_attribute,
                               is_continuous=attribute_is_continuous,
                               continuous_split_value=best_split_value,
                               sub_trees={})

        if attribute_is_continuous:
            # 使用连续值属性划分子树
            split_less = '<{}'.format(best_split_value)
            split_more_or_equal = '>={}'.format(best_split_value)
            df_less = df[df.loc[:, best_split_attribute] < best_split_value].copy()
            df_more_or_equal = df[df.loc[:, best_split_attribute] >= best_split_value].copy()

            df_less.drop(best_split_attribute, axis=1, inplace=True)
            df_more_or_equal.drop(best_split_attribute, axis=1, inplace=True)

            tree.sub_trees[split_less] \
                = ID3DecisionTree.generate_decision_tree_with_continuous_attribute(df_less)
            tree.sub_trees[split_more_or_equal] \
                = ID3DecisionTree.generate_decision_tree_with_continuous_attribute(df_more_or_equal)
        else:
            # 使用离散值属性划分
            best_split_attribute_values = set(df.loc[:, best_split_attribute])
            for attribute_value in best_split_attribute_values:
                # 挑选出原数据集中在属性best_split_attribute取值为attribute_value的子df
                sub_df = df[df.loc[:, best_split_attribute] == attribute_value]

                # 选择的子df需要去掉best_split_attribute这一列
                sub_df_drop_attribute = sub_df.drop(best_split_attribute, axis=1)

                # 递归生成子树
                tree.sub_trees[attribute_value] \
                    = ID3DecisionTree.generate_decision_tree_without_continuous_attribute(sub_df_drop_attribute)

        return tree

    @staticmethod
    def generate_decision_tree_without_continuous_attribute(df: pd.DataFrame):
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
        if len(attributes) == 0 or (len(attributes == 1) and len(set(df.iloc[:, -1])) == 1):
            max_count_label = pd.value_counts(df.iloc[:, -1]).index[0]
            return ID3DecisionTree(is_leaf=True,
                                   sub_trees=max_count_label)

        # 从attributes中选择最佳的划分属性
        best_split_attribute = None
        max_entropy_gain = -np.inf
        for attribute in attributes:
            cur_entropy_gain = ID3DecisionTree.get_discontinuous_entropy_gain(df, attribute)

            if cur_entropy_gain > max_entropy_gain:
                best_split_attribute = attribute
                max_entropy_gain = cur_entropy_gain
            # print('{}: {}'.format(attribute, cur_entropy_gain))

        # 开始划分树节点
        tree = ID3DecisionTree(is_leaf=False,
                               attribute=best_split_attribute,
                               sub_trees={})
        best_split_attribute_values = set(df.loc[:, best_split_attribute])
        for attribute_value in best_split_attribute_values:
            # 挑选出原数据集中在属性best_split_attribute取值为attribute_value的子df
            sub_df = df[df.loc[:, best_split_attribute] == attribute_value]

            # 选择的子df需要去掉best_split_attribute这一列
            sub_df_drop_attribute = sub_df.drop(best_split_attribute, axis=1)

            # 递归生成子树
            tree.sub_trees[attribute_value] \
                = ID3DecisionTree.generate_decision_tree_without_continuous_attribute(sub_df_drop_attribute)
        return tree


if __name__ == '__main__':
    df = pd.read_csv('../../data/chapter_4/watermelon_dataset.csv')

    # 使用不含连续值属性的数据集生成决策树
    df_has_not_continuous_attributes = df.drop(['编号', '密度', '含糖率'], axis=1)
    tree_has_not_continuous_attributes = ID3DecisionTree.generate_decision_tree(df=df_has_not_continuous_attributes,
                                                                                has_continuous_attribute=False)

    # 使用包含连续值属性的数据集生成决策树
    df_has_continue_attributes = df.drop(['编号'], axis=1)
    tree_has_continuous_attributes = ID3DecisionTree.generate_decision_tree(df=df_has_continue_attributes)

