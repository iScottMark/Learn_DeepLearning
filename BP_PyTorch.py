"""
@Time:     2020/4/10 22:19
@Author:   Scott Mark
@File:     BP_PyTorch_test.py
@Software: PyCharm
"""
# -*- coding: utf-8 -*-


from sklearn.datasets import load_boston
from sklearn import preprocessing
import torch

dtype = torch.cuda.FloatTensor

# 载入数据，并预处理
X, y = load_boston(return_X_y=True)
X = preprocessing.scale(X[:100, :])
y = preprocessing.scale(y[:100].reshape(-1, 1))

# 定义超参数
data_size = X.shape[0]      # 共有506个训练集
D_input = X.shape[1]        # 13个输入变量，13个神经元节点
D_output = 1                # 1个输出变量，做回归
D_hidden = 50               # 中间只有一个隐藏层，设置50个神经元节点
lr = 1e-5                   # 设置学习率
epoch = 200000              # 设置训练epoch
