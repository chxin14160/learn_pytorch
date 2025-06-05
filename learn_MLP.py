import torch
from torch import nn
import common
# import matplotlib
# # 强制使用 TkAgg 或 Qt5Agg 后端 (使用独立后端渲染)
# matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'，根据你的系统安装情况
# # matplotlib.use('Qt5Agg')  # 或者使用 'Qt5Agg'，根据你的系统安装情况
import matplotlib.pyplot as plt

# 绘图函数
def plot(x, y, xlabel, ylabel, figsize=(5, 2.5)):
    plt.figure(figsize=figsize)
    plt.plot(x.numpy(), y.numpy(), 'b-', linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


# 生成从-8.0到7.9（不包括8.0）的列表，步长为0.1，需要梯度
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

def learn_ActivationFunction():
    ''' 绘制ReLU函数 '''
    # 通过ReLU函数激活（将所有负值置为0，正值保持不变）
    y = torch.relu(x)
    # detach() 用于从计算图中分离张量，以便转换为numpy数组进行绘图
    # plot(x.detach(), y.detach(), 'x', 'ReLU(x)', figsize=(5, 2.5))
    # 绘图展示了ReLU函数的典型形状：在x<0时输出0，x≥0时输出x

    # 绘制 ReLU函数的导数
    # 执行反向传播
    # torch.ones_like(x)创建一个与x形状相同的全1张量，作为反向传播的梯度输入。假设y的输出对后续计算的贡献都是1
    # retain_graph=True 在反向传播后保留计算图，以便后续还可再进行反向传播。在多次反向传播或计算高阶导数时很有用
    y.backward(torch.ones_like(x), retain_graph=True)

    # 绘图  x.grad是x的梯度，通过反向传播计算得到
    # plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))


    ''' 绘制sigmoid函数 '''
    y = torch.sigmoid(x)
    # plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

    # 绘制 sigmoid函数的导数
    x.grad.data.zero_() # 清除以前的梯度
    y.backward(torch.ones_like(x),retain_graph=True)
    # plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))


    ''' 绘制tanh函数 '''
    y = torch.tanh(x)
    plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

    # 绘制 tanh函数的导数
    x.grad.data.zero_() # 清除以前的梯度
    y.backward(torch.ones_like(x),retain_graph=True)
    plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

# learn_ActivationFunction()

def visual_ActivationFunction():
    ''' 所有激活函数及其导数图像画在一个图里 '''
    plt.figure(figsize=(10, 5))  # 创建一个新的图形对象

    # 添加rule及其导数
    plt.subplot(231)
    y = torch.relu(x)
    plt.plot(x.detach().numpy(), y.detach().numpy(), 'b-', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('ReLU(x)')
    plt.grid(True)

    plt.subplot(234)
    x.grad.data.zero_() # 清除以前的梯度
    y.backward(torch.ones_like(x), retain_graph=True)
    plt.plot(x.detach().numpy(), x.grad.numpy(), 'b-', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('grad of relu')
    plt.grid(True)


    # 添加sigmoid及其导数
    plt.subplot(232)
    y = torch.sigmoid(x)
    plt.plot(x.detach().numpy(), y.detach().numpy(), 'b-', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.grid(True)

    plt.subplot(235)
    x.grad.data.zero_() # 清除以前的梯度
    y.backward(torch.ones_like(x), retain_graph=True)
    plt.plot(x.detach().numpy(), x.grad.numpy(), 'b-', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('grad of sigmoid')
    plt.grid(True)


    # 添加tanh及其导数
    plt.subplot(233)
    y = torch.tanh(x)
    plt.plot(x.detach().numpy(), y.detach().numpy(), 'b-', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('tanh(x)')
    plt.grid(True)

    plt.subplot(236)
    x.grad.data.zero_() # 清除以前的梯度
    y.backward(torch.ones_like(x), retain_graph=True)
    plt.plot(x.detach().numpy(), x.grad.numpy(), 'b-', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('grad of tanh')
    plt.grid(True)

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图形区域
    plt.show()

# visual_ActivationFunction()


''' 多层感知机的从零开始实现 开始 '''
# batch_size = 256
#
# # 输入层有748个输入(28×28的图像展平)，输出层有10个输出(10个类别)，隐藏层256个神经元
# num_inputs, num_outputs, num_hiddens = 784, 10, 256
#
# # 定义参数w1，w2，b1，b2（权重与偏置初始化），皆需计算梯度
# # w1，w2为满足标准正态分布的随机数字，[*0.01进行缩放是因为小随机数有助于训练稳定(打破对称性)]
# # 偏置初始化为 0
# # 使用nn.Parameter包装以便PyTorch跟踪梯度
# W1 = nn.Parameter(torch.randn(
#     num_inputs, num_hiddens, requires_grad=True) * 0.01)
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
#
# W2 = nn.Parameter(torch.randn(
#     num_hiddens, num_outputs, requires_grad=True) * 0.01)
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
#
# params = [W1, b1, W2, b2] # 所有可训练参数放在一个列表中，便于后续操作（如传递给优化器）
''' 多层感知机的从零开始实现 结束 '''


''' 多项式回归 开始 '''
# import math
# import numpy as np
from common import Accumulator
# import common
#
#
# # 生成数据集
# max_degree = 20  # 多项式的最大阶数
# n_train, n_test = 100, 100  # 训练和测试数据集大小
#
# # 真实权重初始化
# true_w = np.zeros(max_degree)  # 分配大量的空间
# true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
#
# # 特征生成
# features = np.random.normal(size=(n_train + n_test, 1)) # 生成200个服从标准正态分布的随机数作为原始特征
# np.random.shuffle(features) # 打乱特征以避免任何潜在的顺序偏差
#
# # 多项式特征扩展
# poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) # 将每个特征扩展为0到19阶的多项式特征
# # np.arange(max_degree): 生成一个从0到max_degree-1的整数数组。(max_degree=20，则生成[0,1,2,...,19])
# # .reshape(1, -1): 将一维数组重塑为一个行向量（1行，max_degree列）。(例如，[0,1,2,...,19]变为[[0,1,2,...,19]] )
# # np.power(features, ...): 对features中的每个元素，计算其对应阶数的幂。(features的形状假设为(n_samples, 1)（例如，(200, 1)）)
# # np.power通过广播机制会将features的每一行与[0, 1, 2, ..., 19]进行逐元素幂运算
#
# # 特征缩放
# for i in range(max_degree): # 对每个多项式特征进行归一化，除以阶乘(gamma函数)
#     # 以防止高阶项的值过大
#     poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
#
# # 标签生成
# # labels的维度:(n_train+n_test,) # 通过多项式特征和真实权重的线性组合生成标签
# labels = np.dot(poly_features, true_w)
# labels += np.random.normal(scale=0.1, size=labels.shape) # 添加高斯噪声(标准差0.1)使数据更真实
#
# # NumPy ndarray转换为tensor (使用列表推导式)
# true_w, features, poly_features, labels = [torch.tensor(x, dtype=
#     torch.float32) for x in [true_w, features, poly_features, labels]]
#
# print(f"查看前两个样本的原始特征：\n{features[:2]}")
# print(f"查看前两个样本的多项式特征（所有阶数）：\n{poly_features[:2, :]}")
# print(f"查看前两个样本的标签：{labels[:2]}")
#
# 对模型进行训练和测试
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)             # 模型预测输出结果
        y = y.reshape(out.shape) # 将实际标签y的形状调整为与模型输出out一致
        l = loss(out, y)         # 模型输出out与实际标签y之间的损失
        metric.add(l.sum(), l.numel()) # 将损失总和 和 样本总数 累加到metric中
    return metric[0] / metric[1] # 损失总和/预测总数，即平均损失
#
# # 训练函数
# def train(train_features, test_features, train_labels, test_labels,
#           num_epochs=400):
#     loss = nn.MSELoss(reduction='none') # 使用均方误差（MSE）作为损失函数,reduction='none'表示不进行任何归约(即返回每个样本的损失，而不是平均值或总和)
#     input_shape = train_features.shape[-1]
#     # 不设置偏置，因为我们已经在多项式中实现了它
#     net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) # 输入维度为input_shape，输出维度为1，不设置偏置项(因为多项式特征中已经包含了常数项(0阶项))
#     batch_size = min(10, train_labels.shape[0]) # 批量大小，最大为10，或训练集样本数（取较小值）
#     train_iter = common.load_array((train_features, train_labels.reshape(-1,1)),
#                                 batch_size) # 将数据加载为可迭代的批量数据
#     test_iter = common.load_array((test_features, test_labels.reshape(-1,1)),
#                                batch_size, is_train=False) # is_train=False表示测试集不需要打乱数据
#     trainer = torch.optim.SGD(net.parameters(), lr=0.01) # 使用随机梯度下降(SGD)优化器，学习率为0.01。优化目标是net的所有参数
#     # 绘制训练过程中的损失曲线
#     animator = common.Animator(xlabel='epoch', ylabel='loss', yscale='log', # yscale='log'：使用对数刻度显示损失值
#                             xlim=[1, num_epochs], ylim=[1e-3, 1e2], # 设置坐标轴范围
#                             legend=['train', 'test'])  # 图例标签
#     for epoch in range(num_epochs): # 循环训练
#         common.train_epoch_ch3(net, train_iter, loss, trainer)
#         if epoch == 0 or (epoch + 1) % 20 == 0:
#             animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
#                                      evaluate_loss(net, test_iter, loss)))
#             # plt.show() # pycharm社区版没有科学模块，通过在循环里show来实现动画效果
#     animator.close()  # 最后记得关闭图形
#     print('训练后的模型权重（即多项式回归的系数）weight:\n', net[0].weight.data.numpy())
#
# # poly_features[:n_train, :4], 前4列，前100行是训练集数据
# # poly_features[n_train:, :4], 前4列，后100行是测试集数据
#
# # 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
# train(poly_features[:n_train, :4], poly_features[n_train:, :4],
#       labels[:n_train], labels[n_train:])
#
# # 从多项式特征中选择前2个维度，即1和x
# train(poly_features[:n_train, :2], poly_features[n_train:, :2],
#       labels[:n_train], labels[n_train:])
#
# # 从多项式特征中选取所有维度
# train(poly_features[:n_train, :], poly_features[n_train:, :],
#       labels[:n_train], labels[n_train:], num_epochs=1500)
''' 多项式回归 结束 '''



''' 权重衰减 开始 '''
# 生成数据
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5 # 训练集数量，测试集数量，输入特征数量，批次大小
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05 # 真实权重，真实偏置

train_data = common.synthetic_data(true_w, true_b, n_train) # 生成训练集数据
train_iter = common.load_array(train_data, batch_size)      # 训练集数据迭代器

test_data = common.synthetic_data(true_w, true_b, n_test)   # 生成测试集数据
test_iter = common.load_array(test_data, batch_size, is_train=False) # 测试集数据迭代器

# 初始化模型参数：权重和偏置
def init_params():
    # 在均值为0，标准差为1的正态分布中随机取值 初始化模型权重，并启用梯度跟踪，以便进行反向传播
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True) # 创建偏置标量
    return [w, b] # 返回的是列表

# 定义L2范数惩罚
def l2_penalty(w): # 这里用最方便的 对所有项求平方后并将它们求和
    return torch.sum(w.pow(2)) / 2 # 所有项平方后求和，/2是为了求导后的形式简洁

# 定义损失函数
def squared_loss(y_hat, y):  #@save y_hat预测值, y真实值
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # /2只是为了与导数的因子2抵消，即为了导数计算更简洁

''' 定义优化算法
# 实现小批量随机梯度下降（Stochastic Gradient Descent, SGD）优化算法
    params: 需更新的参数列表。通常为神经网络的可训练权重w和偏置b
        lr: 学习率（learning rate），是一个标量，用于控制每次参数更新的步长
batch_size: 批量大小，用于调整梯度更新的幅度
'''
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad(): # 禁用梯度计算，所有的操作都不会被记录到计算图中，因此不会影响自动微分的过程。参数更新操作时必须的，因为参数更新本身不应该被微分
        for param in params:
            # 计算参数更新的步长， /batch_size 是为了对小批量数据的梯度进行平均，梯度缩放
            param -= lr * param.grad / batch_size # (param -= ...是将计算出的更新步长应用到参数上，从而更新参数）
            param.grad.zero_() # 将参数的梯度手动清零(因为梯度是累积的，以免影响下一次的梯度计算)

# 训练模型
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: torch.matmul(X, w) + b, squared_loss # 直接使用线性回归模型
    num_epochs, lr = 100, 0.003 # 训练轮数，学习率
    animator = common.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs): # 每轮数据
        for X, y in train_iter:
            # 增加了L2范数惩罚项，即 包括L2正则化
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w) # 正则化项：lambd * l2_penalty(w)（权重L2范数的惩罚）
            l.sum().backward() # 对总损失求和后执行反向传播
            sgd([w, b], lr, batch_size) # 使用随机梯度下降(SGD)进行参数更新
        if (epoch + 1) % 5 == 0: # 每5轮重绘制一次
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), # 训练集上的损失
                                     evaluate_loss(net, test_iter, loss)))    # 测试集上的损失
    print('w的L2范数是：', torch.norm(w).item())

# train(lambd=0)
# train(lambd=3)

# 权重衰减的简洁实现
def train_concise(wd): # wd: 权重衰减（weight decay）系数，相当于L2正则化系数
    net = nn.Sequential(nn.Linear(num_inputs, 1)) # 只有一个全连接层(线性回归模型)
    # 参数初始化
    for param in net.parameters():
        param.data.normal_() # 全部参数皆用.normal_()初始化为正态分布随机数
    # 定义损失函数
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003 # 训练轮数，学习率
    # 偏置参数没有衰减（参数分组设置如下）
    # 对权重 net[0].weight 应用权重衰减（L2正则化）
    # 对偏置 net[0].bias 不应用权重衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = common.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad() # 清除之前的梯度
            l = loss(net(X), y) # 当前批次的损失
            l.mean().backward() # 计算平均损失并反向传播
            trainer.step()      # 执行参数更新
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (evaluate_loss(net, train_iter, loss),
                          evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

# train_concise(0)
# train_concise(3)
''' 权重衰减 结束 '''










