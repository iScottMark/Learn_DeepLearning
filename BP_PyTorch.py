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
X = preprocessing.scale(X[:100, :]) # scale方法是做归一化处理
y = preprocessing.scale(y[:100].reshape(-1, 1))

# 定义超参数
data_size = X.shape[0]      # 共有506个训练集
D_input = X.shape[1]        # 13个输入变量，13个神经元节点
D_output = 1                # 1个输出变量，做回归
D_hidden = 50               # 中间只有一个隐藏层，设置50个神经元节点
lr = 1e-5                   # 设置学习率
epoch = 200000              # 设置训练epoch

# 转换为Tensor
# X = torch.Tensor(X).type(dtype)
# y = torch.Tensor(y).type(dtype)
X = torch.from_numpy(X).type(dtype)
y = torch.from_numpy(y).type(dtype)

# 定义训练参数
w1 = torch.randn(D_input, D_hidden).type(dtype)
w2 = torch.randn(D_hidden, D_output).type(dtype)

# 进行训练
for i in range(epoch):

    # 前向传播
    h = torch.mm(X, w1) # 计算隐层
    h_relu = h.clamp(min=0) # relu
    # y_pred = torch.mm(h_relu, w2) # 输出层
    y_pred = h_relu.mm(w2) # 输出层

    # loss计算，使用L2损失函数
    loss = (y_pred - y).pow(2).sum()

    if i % 10000 == 0:
        print('epoch: {} loss: {:.4f}'.format(i, loss))

    # 反向传播，计算梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = torch.mm(h_relu.t(), grad_y_pred)
    grad_h_relu = torch.mm(grad_y_pred, w2.t())

    # relu函数的倒数，左半段=0，右半段=1
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0

    grad_w1 = torch.mm(X.t(), grad_h)

    # 更新计算的梯度
    w1 = w1 - lr * grad_w1
    w2 = w2 - lr * grad_w2