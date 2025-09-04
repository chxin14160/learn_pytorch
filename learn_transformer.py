import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import common

# print(f"10x10 的单位矩阵（对角线为 1，其余为 0）：\n{torch.eye(10)}")

# 仅当查询和键相同时，注意力权重为1，否则为0
# torch.eye(10): 生成一个 10x10 的单位矩阵（对角线为 1，其余为 0）
# .reshape((1,1,10,10)): 调整形状为(batch_size=1,num_heads=1,seq_len=10,seq_len=10)
# 模拟一个单头注意力机制的权重矩阵（Queries × Keys）
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
common.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')



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

def average_aggregation():
    '''👉 平均汇聚'''
    # .repeat_interleave() 用于重复张量元素
    # 计算出训练样本标签y的均值，然后将其重复n_test次，返回一个长度为n_test的 1D张量
    # 即，生成一个长度为 n_test 的张量，所有元素为 y_train的均值，即 y_train.mean()
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    # plot_kernel_reg(y_hat)
    common.plot_kernel_reg(x_test, y_truth, y_hat, x_train, y_train)
# average_aggregation()

def nonParametric_attention_aggregation():
    '''👉 非参数注意力汇聚'''
    # 1. 重复x_test以匹配注意力权重的形状
    # 为每个测试样本生成一个与 x_train 对齐的矩阵，便于后续计算相似度
    # X_repeat的形状:(n_test,n_train),
    # 每一行都包含着相同的测试输入（例如：同样的查询）
    # x_test.repeat_interleave(n_train) 对张量x_test的每个元素沿指定维度(默认0)重复n_train次
    # 即 每个测试样本重复 n_train 次（展平为一维）
    # .reshape((-1, n_train)) 将展平的张量重新调整为(n_test, n_train)，其中每一行是x_test的一个副本 重复n_train次
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    print(f"测试数据形状x_test.shape：{x_test.shape}")
    print(f"将测试数据元素重复后形状 x_test.repeat_interleave(n_train).shape："
          f"{x_test.repeat_interleave(n_train).shape}")
    print(f"重塑形状为(n_test,n_train)以便后续计算相似度（相似度越高，注意力权重越大）\n"
          f"x_test.repeat_interleave(n_train).reshape((-1, n_train)).shape："
          f"{x_test.repeat_interleave(n_train).reshape((-1, n_train)).shape}")

    # 2. 计算注意力权重（高斯核软注意力）
    # x_train包含着键。attention_weights的形状：(n_test,n_train),
    # 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
    # -(X_repeat - x_train)**2 / 2 计算测试样本与训练样本之间的负欧氏距离（高斯核的指数部分）
    # 测试数据的输入X_repeat 相当于 查询
    # 训练数据的输入x_train  相当于 键
    attention_weights = nn.functional.softmax(
        -(X_repeat - x_train)**2 / 2, # 高斯核(负欧氏距离)，相似度计算：距离越小，相似度越高（权重越大）
        dim=1) # 对每一行（每个测试样本）做softmax归一化，确保权重和为1

    # 3. 加权平均得到预测值
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
    # 左：每个测试样本 对应所有训练标签的各个注意力权重
    # 右：每个训练样本对应的标签
    y_hat = torch.matmul(attention_weights, y_train) # 矩阵乘法：左找行，右找列
    plot_kernel_reg(y_hat)

    # np.expand_dims(attention_weights, 0) 在第0轴(最外层)扩展新维度
    # np.expand_dims(np.expand_dims(attention_weights,0),0) 连续两次在第0轴(最外层)扩展新维度
    # 假设attention_weights原始维度是(3,4)，则第一次扩展变成(1,3,4)，则第一次扩展变成(1,1,3,4)
    common.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs') # 显示注意力权重的矩阵热图
# nonParametric_attention_aggregation()

# def parametric_attention_aggregation():
# 👉 参数注意力汇聚

















