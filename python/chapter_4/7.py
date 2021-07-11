"""
图4.2是一个递归算法，若面临巨量数据，则决策树的层数会很深，使用递归方法易导致"栈"溢出。
试使用"队列"数据结构，以参数MaxDepth控制树的最大深度，写出与图4.2等价、但不使用递归的决策树
生成算法。

输入:
    训练集D = {(x1, y1), (x2, y2), ..., (xm, ym)};
    属性集A = {a1, a2, ..., ad}
    树的最大深度MaxDepth
过程:函数TreeGenerate(D, A, MaxDepth)
初始化当前深度curDepth = 0;
初始化队列节点queue
生成决策树的根节点root
创建决策树节点与数据集D和A的映射tree_to_data = {root: (D, A)}
将root放入队列中
while queue不为空 and curDepth <= MaxDepth:
    for index in range(len(queue)):
        tree_node = queue.front()
        queue.pop()
        sub_data, sub_attribute = tree_to_data[tree_node]
        如果curDepth == MaxDepth:
            根据sub_data中的数据选择样本最多的类别作为节点标签
            continue;
        如果sub_data中的样本类别一致：
            使用样本类别作为节点标签
            continue;
        如果sub_attribute为空集或者sub_data在sub_attribute上的取值相同：
            统计sub_data中数量最多的类别作为节点标签
            continue

        选择最佳的分裂属性best_split_attribute
        获取sub_data在best_split_attribute上的取值结合best_split_attribute_values
        for best_split_attribute_value in best_split_attribute_values:
            生成一个子节点sub_tree
            tree_node.sub_trees[best_split_attribute_value] = sub_tree
            根据取值生成对应的子数据集s_d1和子属性集s_a1
            tree_to_data = {sub_tree: (s_d1, s_a1)
            queue.push(sub_tree)
return tree
"""