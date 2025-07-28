import torch
from matplotlib.lines import lineStyles
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


# 验证 梯度消失
def GradientVanishing():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    y.backward(torch.ones_like(x)) # 用于触发反向传播，计算 y 关于 x 的梯度
    plt.figure(figsize=(4.5,2.5))
    plt.plot(x.detach().numpy(),y.detach().numpy(),label='sigmoid')
    plt.plot(x.detach().numpy(),x.grad.numpy(),linestyle='--',label='gradient')
    plt.xlabel('x')
    plt.ylabel('sigmoid')
    plt.legend() # 添加图例(线的标签)
    plt.grid()   # 启用网格线
    plt.show()

# 验证 梯度爆炸
def GradientExplosion():
    # 梯度爆炸
    # 生成100个高斯随机矩阵，并将其于初始矩阵相乘
    M = torch.normal(0, 1,size=(4,4)) # 从均值=0，标准差=1的正态分布中随机取数生成4*4的矩阵
    print(f"初始矩阵：\n{M}")
    for i in range(100):
        M = torch.mm(M, torch.normal(0, 1, size=(4,4)))
    print(f"乘以100个随机矩阵后\n{M}")

# GradientVanishing()
# GradientExplosion()


''' 多层感知机的从零开始实现 开始 '''
batch_size = 256

# 返回训练集和测试集的迭代器
# load_data_fashion_mnist函数是在图像分类数据集中定义的一个函数，可以返回batch_size大小的训练数据集和测试数据集
train_iter, test_iter = common.load_data_fashion_mnist(batch_size)

# 输入层有748个输入(28×28的图像展平)，输出层有10个输出(10个类别)，隐藏层256个神经元
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 定义参数w1，w2，b1，b2（权重与偏置初始化），皆需计算梯度
# w1，w2为满足标准正态分布的随机数字，[*0.01进行缩放是因为小随机数有助于训练稳定(打破对称性)]
# 偏置初始化为 0
# 使用nn.Parameter包装以便PyTorch跟踪梯度
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2] # 所有可训练参数放在一个列表中，便于后续操作（如传递给优化器）

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

'''
共3层：
输入层：x,无需实现
隐藏层：y=w1*x+b1
隐藏层到输出层前，数据先经过激活函数
隐藏层到输出层：y=w2x+b2
'''
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

# reduction='none' 直接返回 n分样本的loss
loss = nn.CrossEntropyLoss(reduction='none') # 创建一个交叉熵损失函数

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    animator = common.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    # num_epochs：训练次数
    for epoch in range(num_epochs):
        # train_epoch_ch3：训练模型，返回准确率和错误度
        train_metrics = common.train_epoch_ch3(net, train_iter, loss, updater)

        # 在测试数据集上评估精度
        test_acc = common.evaluate_accuracy(net, test_iter)

        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc # 简化链式比较
    assert test_acc <= 1 and test_acc > 0.7, test_acc

num_epochs, lr = 10, 0.1 # 迭代周期数设为10，学习率设为0.1。
updater = torch.optim.SGD(params, lr=lr)
# train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
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


''' 暂退法Dropout 开始 '''
def dropout_layer(X, dropout): # 实现随机失活层 Dropout层
    assert 0 <= dropout <= 1 # 确保dropout在合理范围内，若非，程序会抛出AssertionError，防止非法输入
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X) # 直接返回一个与输入X形状相同的全零张量
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    # 随机生成掩码，并根据掩码对输入X进行部分置零（丢弃部分神经元）
    # (torch.rand(X.shape) > dropout)：生成一个布尔掩码，
    # 其中每个元素以dropout的概率为False（被丢弃），以1 - dropout的概率为True（被保留）
    # .float()：将布尔掩码转换为浮点数掩码（True为1.0，False为0.0）
    mask = (torch.rand(X.shape) > dropout).float()
    # mask * X：将掩码应用到输入X上，被丢弃的神经元（对应掩码为0）输出为0，保留的神经元（对应掩码为1）输出原始值
    # / (1.0 - dropout)：对保留的神经元输出值进行缩放，确保输出的期望值与未应用Dropout时一致
    return mask * X / (1.0 - dropout)

# 测试Dropout层
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(f"不丢弃任何元素，返回原始输入：\n{dropout_layer(X, 0.)}")
print(f"随机丢弃约50%的元素：\n{dropout_layer(X, 0.5)}")
print(f"丢弃所有元素，返回全零张量：\n{dropout_layer(X, 1.)}")

# 输入/输出层的神经元数量，第一和第二个隐藏层的神经元数量
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5 # 两个失活概率，第一个隐藏层后丢弃20%的神经元

# 输入层 →
# 第一个隐藏层（ReLU激活） →
# Dropout →
# 第二个隐藏层（ReLU激活） →
# Dropout →
# 输出层
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__() # 调用父类nn.Module的初始化方法
        self.num_inputs = num_inputs
        self.training = is_training # 表示网络是否处于训练模式（默认为True）
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)     # 第一层隐藏层
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)   # 第二层隐藏层
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)    # 输出层
        self.relu = nn.ReLU()   # 定义ReLU激活函数，用于引入非线性

    def forward(self, X): # 前向传播
        # X.reshape((-1, self.num_inputs))：将输入X展平为形状为(batch_size, num_inputs)的张量
        # -1是占位符，表示“自动计算这一维度的值”。PyTorch会根据 X 的总元素数量和 self.num_inputs 的值自动推断出 batch_size
        # self.lin1(...)：通过第一个全连接层。
        # self.relu(...)：应用ReLU激活函数
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True: H1 = dropout_layer(H1, dropout1) # 在第一个全连接层后添加dropout层
        H2 = self.relu(self.lin2(H1)) # 第二个全连接层
        if self.training == True: H2 = dropout_layer(H2, dropout2) # 在第二个全连接层后添加dropout层
        out = self.lin3(H2) # 输出层
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2) # 初始化网络


num_epochs, lr, batch_size = 10, 0.5, 256       # 训练轮数，学习率，批次大小
loss = nn.CrossEntropyLoss(reduction='none')    # 损失函数使用交叉熵，reduction='none'表示不对损失值进行求和或平均
train_iter, test_iter = common.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
# train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


# 暂退法Dropout的简洁实现
# 与从零开始实现的模型结构一样：
# 展平层 →
# 第一层线性层 → relu激活 →
# Dropout →
# 第二层线性层 → relu激活 →
# Dropout →
# 输出层
# nn.Sequential是一个容器，用于按顺序组合多个层。数据会依次通过这些层
# nn.ReLU(): 激活函数，用于引入非线性
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    # 在第一个全连接层之后添加一个dropout层
                    nn.Dropout(dropout1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    # 在第二个全连接层之后添加一个dropout层
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))

def init_weights(m): # 初始化模型权重
    if type(m) == nn.Linear: # 只对全连接层做初始化参数
        nn.init.normal_(m.weight, std=0.01) # 使用正态分布初始化权重（均值=0，标准差=0.01）

# 将init_weights应用于 net 中的每一个子模块。
# apply 方法会递归地遍历整个网络，并对每个模块调用 init_weights 函数
net.apply(init_weights) # 递归应用初始化函数到所有层

# torch.optim.SGD随机梯度下降优化器。SGD 是常用的优化算法，用于在训练过程中更新神经网络的参数，以最小化损失函数
# net.parameters() 返回模型中所有可训练参数（权重和偏置）的生成器。这些参数将在训练过程中被优化器更新
trainer = torch.optim.SGD(net.parameters(), lr=lr)
# train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
''' 暂退法Dropout 结束 '''







