from common import Timer
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import random

''' 矢量化加速例子 开始 '''
# # 实例化两个全为1的10000维向量
# n = 10000
# a = torch.ones([n])
# b = torch.ones([n])
#
# # 开始对工作负载进行基准测试
# c = torch.zeros(n)
# timer = Timer()
#
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f"方法一：使用循环遍历向量，耗时：{timer.stop():.5f} sec")
#
# timer.start()
# d = a + b
# print(f"方法二：使用重载的+运算符来计算按元素的和，耗时：{timer.stop():.5f} sec")
''' 矢量化加速例子 结束 '''


''' 正态分布与平方缺失 开始 '''
# # 计算正态分布
# def normal(x, mu, sigma):
#     p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
#     return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
#
# # 可视化不同参数的正态分布（高斯分布）概率密度函数
# x = np.arange(-7, 7, 0.01) # 生成数据点(即 创建从-7到7（步长0.01）的等间距数组，共1400个点，作为横坐标
# params = [(0, 1), (0, 2), (3, 1)] # 三个正态分布的参数对(均值mu，标准差sigma)
# # 第一个：μ=0, σ=1（标准正态分布）
# # 第二个：μ=0, σ=2（更宽的分布）
# # 第三个：μ=3, σ=1（向右平移的分布）
#
# # 可视化
# plt.figure(figsize=(4.5, 2.5))
# for mu,sigma in params:
#     plt.plot(x, normal(x, mu, sigma), label=f'mean{mu}, std{sigma}')
#
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.xlabel('x轴')
# plt.ylabel('p(x)')
# plt.legend()
# plt.grid() # 添加网格
# plt.show()
''' 正态分布与平方缺失 结束 '''


''' 线性回归的从零开始实现 开始 '''
# 生成数据
# 数据集中有1000个样本，每个样本包含从标准正态分布中采样的2个特征
'''
# 生成 ”符合线性关系 y=Xw+b+噪声“ 的合成数据集
# w: 权重向量（决定线性关系的斜率）
# b: 偏置项（决定线性关系的截距）
# num_examples: 要生成的样本数量
'''
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    # 生成一个形状为 (num_examples, len(w)) 的矩阵,每个元素从均值为0、标准差为1的正态分布中随机采样
    X = torch.normal(0, 1, (num_examples, len(w)))
    print(f"X的形状{X.shape}")
    y = torch.matmul(X, w) + b          # 计算线性部分 Xw + b
    y += torch.normal(0, 0.01, y.shape) # 添加噪声（均值为0，标准差为0.01的正态分布）使数据更接近真实场景（避免完全线性可分）
    return X, y.reshape((-1, 1))        # 返回特征矩阵X和标签向量y, y.reshape((-1, 1)) 确保y是列向量（形状为 (num_examples, 1)）

# 定义真实的权重 w = [2, -3.4] 和偏置 b = 4.2
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# features: 形状为 (1000, 2) 的矩阵，每行皆包含一个二维数据样本
# labels: 形状为 (1000, 1) 的向量，每行皆包含一维数据标签值(一个标量)
# 标签由线性关系 y = 2*x1 - 3.4*x2 + 4.2 + 噪声 生成
features, labels = synthetic_data(true_w, true_b, 1000) # 生成1000个样本
print('features:', features[0],'\nlabel:', labels[0])

# 注意：由于有两个特征，完整的可视化需要3D图或两个2D子图。
# 生成第二个特征features[:, 1]和labels的散点图，直观观察两者之间的线性关系
# 假设只看第二个特征和标签的关系（忽略第一个特征）
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# features[:, 1].numpy(): 提取所有样本的第二个特征（x2）并将其转换为 NumPy 数组以便绘图
# labels.numpy(): 将标签转换为 NumPy 数组
# plt.scatter: 绘制散点图，其中 1 是点的大小（可以根据需要调整）
plt.xlabel('x2')
plt.ylabel('y')
# plt.show()

'''
from mpl_toolkits.mplot3d import Axes3D
# 绘制3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(), labels.numpy())
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('3D Scatter plot of x1, x2 vs y')
plt.show()
'''


# 读取数据
'''
打乱数据集 中的样本并以小批量方式获取数据，用于训练
输入:
批量大小 batch_size
特征矩阵 features
标签向量 labels
生成大小为batch_size的小批量
'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)        # 特征矩阵中的样本数量
    indices = list(range(num_examples)) # 生成索引列表，从 0 到 num_examples - 1
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)             # 随机打乱索引列表，以便在每个 epoch 中以随机顺序访问数据，有助于提高模型的泛化能力
    for i in range(0, num_examples, batch_size):            # 从 0 开始，以 batch_size 为步长，遍历整个数据集
        batch_indices = np.array(                           # batch_indices 是 NumPy 数组，包含当前小批量的索引
            indices[i: min(i + batch_size, num_examples)])  # min(i + batch_size, num_examples) 确保最后一个批次不会超出数据集的范围
        yield features[batch_indices], labels[batch_indices]    # yield 关键字用于生成一个小批量的特征和标签，每次调用 next() 或在 for 循环中迭代时，生成器会返回一个批次的数据

batch_size = 10
# 读取第一个小批量数据样本并打印
for X, y in data_iter(batch_size, features, labels):
    print('Features:', X.shape, 'Labels:', y.shape)
    print(X, '\n', y) # 每个批量的特征维度显示 批量大小和输入特征数。批量的标签形状同batch_size
    break


# 初始化模型参数(权重和偏置)
# requires_grad=True表示对张量进行自动微分，也就是说，PyTorch会在反向传播过程中自动计算这个张量的梯度
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) # 使用正态分布(高斯分布)初始化一个形状为(2, 1)的张量。其值从均值为0、标准差为0.01 的正态分布中随机抽取
b = torch.zeros(1, requires_grad=True) # 将偏置初始化为0


# 定义模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):  #@save y_hat预测值, y真实值
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # /2只是为了与导数的因子2抵消，即为了导数计算更简洁


# 定义优化算法
'''
# 实现小批量随机梯度下降（Stochastic Gradient Descent, SGD）优化算法
    params: 需更新的参数列表。通常为神经网络的可训练权重w和偏置b
        lr: 学习率（learning rate），是一个标量，用于控制每次参数更新的步长
batch_size: 批量大小，用于调整梯度更新的幅度
'''
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad(): # 禁用梯度计算，所有的操作都不会被记录到计算图中，因此不会影响自动微分的过程。参数更新操作时必须的，因为参数更新本身不应该被微分
        for param in params:
            # 计算参数更新的步长， /batch_size 是为了对小批量数据的梯度进行平均
            param -= lr * param.grad / batch_size # (param -= ...是将计算出的更新步长应用到参数上，从而更新参数）
            param.grad.zero_() # 将参数的梯度手动清零(因为梯度是累积的，以免影响下一次的梯度计算)

# 开始训练
lr = 0.03           # 学习率，用于控制每次参数更新的步长
num_epochs = 3      # 训练的轮数，表示数据集将被完整遍历的次数
net = linreg        # 线性回归模型
loss = squared_loss # 损失函数，这里使用的是均方差损失函数

for epoch in range(num_epochs):
    # 训练循环
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward() # 反向传播和梯度计算，sum()将损失值加总成一个标量后(对损失求和)，backward()自动计算所有参数的梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    # 打印训练损失
    with torch.no_grad(): # 禁用梯度计算，所有的操作都不会被记录到计算图中，因此不会影响自动微分的过程。参数更新操作时必须的，因为参数更新本身不应该被微分
        train_l = loss(net(features, w, b), labels) # 计算整个数据集上的损失
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}') # 打印当前 epoch 的平均损失

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
''' 线性回归的从零开始实现 结束 '''


''' 线性回归的简洁实现 开始 '''
# 生成数据
# 定义真实的权重 w = [2, -3.4] 和偏置 b = 4.2
true_w = torch.tensor([2, -3.4])
true_b = 4.2


''' 线性回归的简洁实现 结束 '''