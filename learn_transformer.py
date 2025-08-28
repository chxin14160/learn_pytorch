import torch
from torch import nn
import matplotlib.pyplot as plt
import common

# print(f"10x10 的单位矩阵（对角线为 1，其余为 0）：\n{torch.eye(10)}")

# 仅当查询和键相同时，注意力权重为1，否则为0
# torch.eye(10): 生成一个 10x10 的单位矩阵（对角线为 1，其余为 0）
# .reshape((1,1,10,10)): 调整形状为(batch_size=1,num_heads=1,seq_len=10,seq_len=10)
# 模拟一个单头注意力机制的权重矩阵（Queries × Keys）
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
# common.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')



n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5) # 排序后的训练样本(以便可视化)

def f(x): # 非线性函数
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出(添加噪声项)
x_test = torch.arange(0, 5, 0.1) # 测试样本
y_truth = f(x_test)              # 测试样本的真实输出
n_test = len(x_test)             # 测试样本数
print(f"测试样本数：{n_test}")

def plot_kernel_reg(y_hat):
    common.plot_kernel_reg(x_test, y_truth, y_hat, x_train, y_train)

# .repeat_interleave() 用于重复张量元素
# 计算出训练样本标签y的均值，然后将其重复n_test次，返回一个长度为n_test的 1D张量
# 即，生成一个长度为 n_test 的张量，所有元素为 y_train 的均值
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# plot_kernel_reg(y_hat)
common.plot_kernel_reg(x_test, y_truth, y_hat, x_train, y_train)







