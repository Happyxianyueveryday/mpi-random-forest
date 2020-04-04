# RandomForestRegressor.py: 随机森林回归器模块

import os, sys, time 
import numpy as np
import math 
from mpi4py import MPI
from DecisionTreeCartRegression import DecisionTreeCartRegression

class RandomForestRegressor:
    '''
    : RandomForestRegressor: 并行化随机森林回归器模块
    '''
    def __init__(self, comm, main=0, n_estimator=10, oob_score=False, max_features=None, min_impurity_decrease=None, max_depth=None, min_samples_leaf=1, min_samples_split=2):
        '''
        : __init__: 随机森林初始化
        : param comm: mpi4py.MPI.Intracomm, 并行化训练使用的MPI通信子
        : param main: int, 随机森林中训练和预测作为主进程的编号
        : param n_estimator: int, Bagging框架参数，随机森林中包含的基分类器/决策树的数量，默认值为10
        : param continuity: bool, Bagging框架参数，指定输入的数据集是连续数据还是离散数据，默认值为True
        : param oob_score: bool, Bagging框架参数，是否使用袋外样本评估模型的性能和表现，若使用袋外样本则会对随机森林进行迭代以找到最优随机森林，默认值为False
        : param max_features: int/float/str, 决策树剪枝参数，决策树每次分裂时需要考虑的特征数量；默认为None，这里保持和sklearn一致，即和RandomForestClassifier不相同，即选取全部特征参与计算；若输入为int，则表示每次分裂考虑的特征具体值；若输入为float，则表示每次分裂考虑的特征比例；若输入为str，'sqrt'表示每次分裂考虑的特征数量为总特征数量的平方根，'log'表示每次分裂考虑的特征数量为总特征数量的以2作为底的对数
        : param min_impurity_decrease: float, 决策树剪枝参数，选择特征进行分枝过程时的信息增益阈值，信息增益小于输入值的结点分枝操作不会发生，默认值为None
        : param max_depth: int, 决策树剪枝参数，决策树的最大深度，默认值为None
        : param min_samples_leaf: int, 决策树剪枝参数，限制一个结点分枝后的子结点中均至少含有的样本数量，不满足条件则分枝不会发生，默认值为1
        : param min_samples_split: int, 决策树剪枝参数，限制一个分枝的结点所至少需要含有的样本数量，默认值为2
        '''
        # 1. Bagging框架参数
        self.n_estimator = n_estimator       # 随机森林训练的决策树数量
        self.oob_score = oob_score           # 是否使用袋外样本评估的标志                   
        # 2. 决策树剪枝参数
        self.max_features = max_features                     # 决策树每次分裂时需要考虑的特征数量
        self.min_impurity_decrease = min_impurity_decrease   # 决策树每次分裂时的样本评估参数（信息增益比，信息增益值，或者基尼系数）的阈值
        self.max_depth = max_depth                           # 决策树的最大深度
        self.min_samples_leaf = min_samples_leaf             # 决策树分枝后的子结点中至少含有的样本数量
        self.min_samples_split = min_samples_split           # 决策树分枝所至少含有的样本数量
        # 3. 基础参数与数据
        self.comm = comm             # 并行训练和预测使用的MPI通信子
        self.main = main             # 主进程编号
        self.rank = comm.rank        # 当前线程编号
        self.size = comm.size        # 所有进程总数
        self.treeroot = []           # 决策树根结点列表
        self.oob_accuracy = []       # 决策树的袋外准确率列表
        self.n_label = 0             # 训练时的标签种类数量

    def fit(self, data=None, label=None):
        '''
        : fit: 使用训练数据集并行化训练随机森林模型
        : param data: np.array, 二维训练数据集，其中行代表样本，列代表特征，主进程需要提供该参数，辅进程则直接使用默认值None
        : param label: np.array, 一维训练标签集，主进程需要提供该参数，辅进程则直接使用默认值None
        '''
        # 1. 主进程将原始的数据集和标签集广播给各个辅进程，由于随机森林可能需要进行迭代，因此由主进程进行有放回随机抽样后再分发给各个辅进程的方式实际上表现不佳
        data = comm.bcast(data, root=self.main)      # 广播发送/接收训练数据集
        label = comm.bcast(label, root=self.main)    # 广播发送/接收训练标签集
        # 2. 进程计算自身需要构造的决策树数量，所有的辅进程均构造floor(self.n_estimator/self.size)个决策树，而主进程构造(self.n_estimator - (size-1)*floor(self.n_estimator/self.size))个决策树
        if self.rank:
            n_tree = math.floor(self.n_estimator/self.size)     # 本进程需要训练的决策树数量
        else:
            n_tree = self.n_estimator - (self.size - 1)*math.floor(self.n_estimator/self.size)
        # 3. 进行训练过程
        n_sample = label.size   # 训练集的样本数量
        self.n_label = np.unique(label).size
        for i in range(n_tree):
            # 3.1. 每个进程从原始数据集和标签集中进行有重复随机抽样，抽样进行n_sample次
            inbag_index = np.unique(np.random.randint(0, n_sample, size=n_sample))   # 有重复随机抽样后再去重得到的袋内样本下标
            if self.oob_score:
                outbag_index = np.setdiff1d(np.arange(n_sample), inbag_index)        # 如果用户指定需要进行袋外预测和评估，则生成对应的袋外样本下标
            # 3.2 按照抽样的样本创建并且训练指定种类的决策树
            tree = DecisionTreeCartRegression()
            tree.fit(data[inbag_index], label[inbag_index], self.min_impurity_decrease, self.max_depth, self.min_samples_leaf, self.min_samples_split, self.max_features)
            self.treeroot.append(tree)
            # 3.3 若用户指定需要袋外准确率数据，则生成对应的袋外样本准确率数据
            if self.oob_score:
                self.oob_accuracy.append(tree.score(data[outbag_index], label[outbag_index]))
        return

    def predict(self, data=None):
        '''
        : predict: 使用测试数据集并行化预测并返回预测结果
        : param data: np.array, 二维测试数据集，其中行代表样本，列代表特征，主进程需要提供该参数，辅进程则直接使用默认值None
        : return: np.array, 一维测试标签集，预测得到的一维测试集标签，主进程返回结果，辅进程返回None
        '''
        # 1. 主进程将测试数据集分发给各个辅进程
        data = comm.bcast(data, root=self.main)      # 广播发送/接收测试数据集
        # 2. 各个进程在本地依次处理测试数据集中的样本
        label = 0
        for tree in self.treeroot:
            label += tree.predict(data)    
        # 3. 主进程从各个进程搜集数据并且统计出最终预测结果
        label = comm.gather(label, root=self.main)
        if self.rank!=self.main:
            return None
        else:
            res = 0
            for x in label:
                res += x
            res /= self.n_estimator
            return res

    def score(self, data=None, label=None):
        '''
        : score: 使用测试数据集并行化预测，并返回R^2值作为评估
        : param data: np.array, 二维测试数据集，其中行代表样本，列代表特征，主进程需要提供该参数，辅进程则直接使用默认值None
        : param label: np.array, 一维测试标签集，主进程需要提供该参数，辅进程则直接使用默认值None
        : return: float, 本次测试的R^2值
        '''
        predict = self.predict(data)
        if self.rank!=self.main:
            return None
        else:
            r2 = 1 - np.sum((label - predict)**2)/np.sum((label - np.average(label))**2)
            return r2

    def get_oob_score(self):
        '''
        : get_oob_score: 返回当前进程所创建的决策树的袋外预测准确率
        : return: np.array: 当前进程所创建的决策树的袋外预测准确率，若用户创建随机森林时给定的参数oob_score值为False，则返回None
        '''
        if self.oob_score:
            return self.oob_accuracy
        else:
            return None

