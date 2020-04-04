# DecisionTreeCartDiscrete.py: 决策树模块

import os, sys, time
import numpy as np
import math
from mpi4py import MPI
from collections import Counter
from scipy import stats 

class TreeNode:
    '''
    : TreeNode: CART决策树结点
    '''
    def __init__(self, _sign, _feature=None, _split=None, _category=None):
        '''
        : __init__: 初始构造函数
        : param _sign: bool, 结点属性标志，值为True表示当前结点为中间结点，值为False表示当前结点为叶子结点
        : param _feature: int, 结点分枝时选择的特征列下标，默认值为None，创建中间结点需要提供该参数，创建叶子结点则无需提供该参数
        : param _split: int, 结点分枝特征的取值
        : param _category: int, 叶子结点的最终判别种类，默认值为None，创建叶子结点需要提供该参数，创建中间结点则无需提供该参数
        '''
        # 1. 初始化结点属性标记
        self.sign = _sign                   # 结点类型标记: True表明结点为中间结点，False表明结点为叶子结点
        # 2. 中间结点参数初始化
        self.left, self.right = None, None  # 中间结点参数: 左子结点和右子结点
        self.feature = _feature             # 中间结点参数: 选择分枝的属性的列下标
        self.split = _split                 # 中间结点参数: 选择分枝属性的取值
        # 3. 叶子结点参数初始化
        self.category = _category           # 叶子结点参数: 叶子结点的最终判别类别


class DecisionTreeCartDiscrete:
    '''
    : DecisionTreeCartDiscrete: 离散CART决策树模块
    '''
    def __init__(self):
        '''
        : __init__: 初始构造函数
        '''
        self.treeroot = None                # 决策树的根结点

    def train(self, data, label, min_impurity_decrease=None, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=None):
        '''
        : train: 训练决策树
        : param data: np.array, 二维训练集
        : param label: np.array, 一维标签集
        : param min_impurity_decrease: float, 预剪枝调整参数，选择特征进行分枝过程时的信息增益阈值，信息增益小于输入值的结点分枝操作不会发生，默认值为None
        : param max_depth: int, 预剪枝调整参数, 决策树的最大深度，默认值为None
        : param min_samples_leaf: int, 预剪枝调整参数，限制一个结点分枝后的子结点中均至少含有的样本数量，不满足条件则分枝不会发生，默认值为1
        : param min_samples_split: int, 预剪枝调整参数，限制一个分枝的结点所至少需要含有的样本数量，默认值为2
        : param max_features: int/float/str, 决策树剪枝参数，决策树每次分裂时需要考虑的特征数量，若输入为int，则表示每次分裂考虑的特征具体值；若输入为float，则表示每次分裂考虑的特征比例；若输入为str，'sqrt'表示每次分裂考虑的特征数量为总特征数量的平方根，'log'表示每次分裂考虑的特征数量为总特征数量的以2作为底的对数
        '''
        n_sample, n_feature = np.shape(data)
        self.treeroot = self._train_cart(data, label, np.full(n_sample, True), 0, min_impurity_decrease, max_depth, min_samples_leaf, min_samples_split, max_features)

    def score(self, data, label):
        '''
        : score: 使用测试集评估准确率
        : param data: np.array, 二维数据集
        : param label: np.array, 一维数据集
        : return: float: 在测试集上评估的准确率
        '''
        res = []
        for sample in data:
            now = self.treeroot
            while now and now.sign:
                if sample[now.feature] == now.split:
                    now = now.left
                else:
                    now = now.right
            res.append(now.category)
        res = np.array(res)
        accuracy = np.sum(res==label)/label.size
        return accuracy

    def predict(self, data):
        '''
        : predict: 对输入的样本进行预测
        : param data: np.array, 二维数据集
        : return: int, 输入样本的标签
        '''
        res = []
        for sample in data:
            now = self.treeroot
            while now and now.sign:
                if sample[now.feature] == now.split:
                    now = now.left
                else:
                    now = now.right
            res.append(now.category)
        res = np.array(res)
        return res

    def print(self):
        '''
        : print: 输出当前训练得到的决策树的层次遍历序列
        '''
        res = self.level_traverse()
        for x in res:
            print("---------------------------------------")
            for y in x:
                print(y)
        print("")

    def level_traverse(self):
        '''
        : level_traverse: 层次遍历整个决策树并且返回遍历序列
        : return: list[list[dict]], 当前决策树的层次遍历序列
        '''
        que = []
        res = []
        if not self.treeroot:
            return res
        que.append((self.treeroot, 0))
        while que:
            now_node, now_level = que[0]
            que.pop(0)
            if not now_node:
                continue
            node_out = {"sign":now_node.sign, "feature":now_node.feature,  "split":now_node.split, "category":now_node.category}
            if now_level>=len(res):
                res.append([node_out])
            else:
                res[now_level].append(node_out)
            if now_node.left:
                que.append((now_node.left, now_level+1))
            if now_node.right:
                que.append((now_node.right, now_level+1))
        return res

    def _train_cart(self, data, label, sample_mask, now_depth, min_impurity_decrease, max_depth, min_samples_leaf, min_samples_split, max_features):
        '''
        : _train_cart: 本方法为私有方法，使用cart算法训练生成决策树
        : param data: np.array, 二维训练集，其中行作为样本，列作为特征
        : param label: np.array, 一维标签集
        : param sample_mask: np.array, 样本掩码向量，用True表示参与本次训练的样本集，False表示不参与
        : param now_depth: int, 当前递归的深度
        : param min_impurity_decrease: float, 预剪枝调整参数，选择特征进行分枝过程时的信息增益阈值，信息增益小于输入值的结点分枝操作不会发生，默认值为None
        : param max_depth: int, 预剪枝调整参数, 决策树的最大深度，默认值为None
        : param min_samples_leaf: int, 预剪枝调整参数，限制一个结点分枝后的子结点中均至少含有的样本数量，不满足条件则分枝不会发生，默认值为1
        : param min_samples_split: int, 预剪枝调整参数，限制一个分枝的结点所至少需要含有的样本数量，默认值为2
        : param max_features: int/float/str, 决策树剪枝参数，决策树每次分裂时需要考虑的特征数量，若输入为int，则表示每次分裂考虑的特征具体值；若输入为float，则表示每次分裂考虑的特征比例；若输入为str，'sqrt'表示每次分裂考虑的特征数量为总特征数量的平方根，'log'表示每次分裂考虑的特征数量为总特征数量的以2作为底的对数
        : return: TreeNode, 训练生成的决策树根节点
        '''
        # 1. 若出现以下的三种情况，则生成叶子结点，且直接将样本集中出现次数最多的标签作为叶子结点类别
        # (1) 参与训练的样本数量小于min_samples_split
        # (2) 参与训练的样本数量小于min_samples_split
        # (3) 若当前结点的深度等于限定的最大深度
        sample_label = label[sample_mask]     # 参与训练的样本的标签
        n_sample = sample_label.size          # 参与训练的样本数量
        if (n_sample<min_samples_split) or (max_depth and now_depth>=max_depth):
            return TreeNode(False, _category=stats.mode(sample_label)[0][0])
        # 2. 若参与训练的样本集中的样本均属于同一类，则创建的决策树为单叶子结点树，并且将该类的标签作为叶子结点的类别
        elif np.unique(sample_label).size==1:
            return TreeNode(False, _category=sample_label[0])
        # 3. 若训练集不为空，计算各个特征的各个取值的基尼系数，选取其中最小的基尼系数，以及取得最小值对应的特征和特征值，然后递归划分生成决策树
        sample_unique = np.unique(sample_label)
        n_feature = np.shape(data)[1]                   # 特征的总数量
        min_gini = float('inf')                         # 最小的基尼指数
        if not max_features:
            feature_index = np.arange(n_feature)        # 若参数max_features为空，则所有的特征均参与计算基尼系数
        elif max_features=='sqrt':
            feature_index = np.random.permutation(n_feature)[:math.floor(math.sqrt(n_feature))]   # 若参数max_features为'sqrt'，则floor(sqrt(n_feature))个特征参与计算基尼系数
        elif max_features=='log':
            feature_index = np.random.permutation(n_feature)[:math.floor(math.log(n_feature, 2))] # 若参数max_features为'log'，则floor(log(n_feature, 2))个特征参与计算基尼系数
        elif isinstance(max_features, int):
            feature_index = np.random.permutation(n_feature)[:max_features]                       # 若参数max_features为整数，则max_features个特征参与计算基尼系数
        elif isinstance(max_features, float):
            feature_index = np.random.permutation(n_feature)[:math.floor(n_feature*max_features)] # 若参数max_features为浮点数，则n_feature*max_feature个特征参与计算基尼系数
        else:
            feature_index = np.arange(n_feature)
        for i in feature_index:
            feature = data[:, i]                        # 当前计算的特征向量列
            sample_feature = feature[sample_mask]       # 参与训练的样本特征取值
            feature_unique = np.unique(sample_feature)
            feature_dic = {x: 0 for x in feature_unique}# 各个特征的取值计数字典                      
            label_dic = {y: 0 for y in sample_unique}   # 各个标签的取值计数字典
            feature_label_dic = {x: {y: 0 for y in sample_unique} for x in feature_unique}  # 特征值等于x的样本的标签计数字典
            for feature_element, label_element in zip(sample_feature, sample_label):
                feature_dic[feature_element] += 1
                label_dic[label_element] += 1
                feature_label_dic[feature_element][label_element] += 1
            for x in feature_dic.keys():
                d1_size, d2_size = feature_dic[x], n_sample-feature_dic[x]    # |D1|, |D2|
                d1_gini, d2_gini = 1, 1    # gini(D1), gini(D2)
                if d1_size and d2_size:
                    for y in feature_label_dic[x].keys():
                        d1_gini -= (feature_label_dic[x][y]/d1_size)**2
                        d2_gini -= ((label_dic[y] - feature_label_dic[x][y])/d2_size)**2
                elif d1_size:
                    for y in feature_label_dic[x].keys():
                        d1_gini -= (feature_label_dic[x][y]/d1_size)**2
                    d2_gini = 0
                elif d2_size:
                    for y in feature_label_dic[x].keys():
                        d2_gini -= ((label_dic[y] - feature_label_dic[x][y])/d2_size)**2
                    d1_gini = 0
                feature_gini = d1_size*d1_gini/n_sample + d2_size*d2_gini/n_sample
                if feature_gini<min_gini:
                    min_gini, min_feature, min_value, left_size, right_size, min_feature_value = feature_gini, i, x, d1_size, d2_size, feature
        if left_size<min_samples_leaf or right_size<min_samples_leaf:
            return TreeNode(False, _category=stats.mode(sample_label)[0][0])
        else:
            temp_mask = np.where(min_feature_value==min_value, True, False)
            sample_mask_left = np.bitwise_and(sample_mask, temp_mask)
            sample_mask_right = np.bitwise_and(sample_mask, np.bitwise_not(temp_mask))
            root = TreeNode(True, _feature=min_feature, _split=min_value) 
            root.left = self._train_cart(data, label, sample_mask_left, now_depth+1, min_impurity_decrease, max_depth, min_samples_leaf, min_samples_split, max_features)
            root.right = self._train_cart(data, label, sample_mask_right, now_depth+1, min_impurity_decrease, max_depth, min_samples_leaf, min_samples_split, max_features)
            return root

