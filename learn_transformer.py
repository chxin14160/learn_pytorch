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
# 👉 带参数注意力汇聚
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(f"批量矩阵乘法bmm后，结果矩阵形状：{torch.bmm(X, Y).shape}")

# 演示注意力机制中的加权求和
weights = torch.ones((2, 10)) * 0.1          # 形状: (批量大小, 序列长度)-注意力权重
values = torch.arange(20.0).reshape((2, 10)) # 形状: (批量大小, 序列长度)-值向量
# .unsqueeze()在指定位置增加维度
print(f"在指定位置增加维度后矩阵形状：\n"
      f"weights：{weights.unsqueeze(1).shape}\n"
      f"values ：{values.unsqueeze(-1).shape}")
print(f"增加维度后 进行 批量矩阵乘法bmm，结果为：\n"
      f"使用bmm计算加权和: 权重(2,1,10) × 值(2,10,1) = 结果(2,1,1) \n"
      f"{torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))}")


class NWKernelRegression(nn.Module):
    ''' Nadaraya-Watson 核回归模型,实现基于注意力机制的核回归 '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 可学习的参数 (高斯核的带宽参数)，即 查询与键间距离要乘以的权重
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        '''
        queries : 查询输入 (n_query, dim)
        keys    : 训练输入 (n_train, dim)
        values  : 训练输出 (n_train, dim)
        # queries 和 attention_weights的形状为 (查询个数，“键－值”对个数)
        '''
        # 扩展 查询向量queries形状 以匹配 键值对keys的维度
        # queries形状: (查询个数,) -> 扩展为 (查询个数, 键值对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # 计算注意力权重（使用高斯核函数）
        # 公式: attention = softmax(-(query - key)^2 * w^2 / 2)
        # 注意力权重通过高斯核 exp(-(x_query-x_key)^2 / (2σ^2)) 计算，这里用softmax归一化
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1) # 形状 (n_query, n_train)
        # 使用注意力权重对值进行加权求和得到预测值
        # bmm: (批量大小, 1, 键值对个数) × (批量大小, 键值对个数, 1) = (批量大小, 1, 1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1), # (n_query, 1, n_train)
                         values.unsqueeze(-1)).reshape(-1) # (n_query, 1, 1) → (n_query,)

# 准备训练时的 keys 和 values
# 生成训练数据的所有组合（用于自注意力）
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
X_tile = x_train.repeat((n_train, 1)) # 形状 (n_train * n_train, dim)
Y_tile = y_train.repeat((n_train, 1)) # 形状 (n_train * n_train, dim)

# 创建键和值（排除对角线元素，即自身  避免自匹配）
# mask 用于排除自匹配（即查询点不与自身计算注意力）
mask = (1 - torch.eye(n_train)).type(torch.bool) # 形状 (n_train, n_train)
# keys的形状  :('n_train'，'n_train'-1)
# values的形状:('n_train'，'n_train'-1)
keys   = X_tile[mask].reshape((n_train, -1)) # 形状 (n_train, n_train-1)
values = Y_tile[mask].reshape((n_train, -1)) # 形状 (n_train, n_train-1)


# 初始化模型、损失函数和优化器
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none') # 逐元素计算损失
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = common.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad() # 清零梯度
    # 计算每个训练点的预测值（queries=x_train, keys/values来自其他点）
    l = loss(net(x_train, keys, values), y_train) # 前向传播
    l.sum().backward()  # 反向传播（对损失求和后反向传播）
    trainer.step()      # 更新参数
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))


# 测试阶段：每个测试点与所有训练点计算注意力
# keys的形状 :(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
# value的形状:(n_test，n_train)
keys   = x_train.repeat((n_test, 1)) # 形状 (n_test, n_train)
values = y_train.repeat((n_test, 1)) # 形状 (n_test, n_train)
y_hat = net(x_test, keys, values).unsqueeze(1).detach() # 预测
plot_kernel_reg(y_hat) # 绘制回归结果

# 可视化注意力权重（测试点 vs 训练点）
# net.attention_weights形状: (n_test, n_train)
# 添加两个维度使其变为(1, 1, n_test, n_train)以匹配show_heatmaps期望的4D输入
common.show_heatmaps(
    net.attention_weights.unsqueeze(0).unsqueeze(0), # 增加批次和头维度
    xlabel='Sorted training inputs',
    ylabel='Sorted testing inputs')
plt.pause(4444)  # 间隔的秒数： 4s




