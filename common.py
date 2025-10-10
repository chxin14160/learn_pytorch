import time
import numpy as np
# from IPython import display
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torchvision import datasets,transforms

# import matplotlib
# # 强制使用 TkAgg 或 Qt5Agg 后端 (使用独立后端渲染)
# matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'，根据你的系统安装情况
# # matplotlib.use('Qt5Agg')  # 或者使用 'Qt5Agg'，根据你的系统安装情况
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'，根据你的系统安装情况

import hashlib
import os
import tarfile
import zipfile
import requests

import collections # 提供高性能的容器数据类型，替代Python的通用容器(如 dict, list, set, tuple)
import re # 供正则表达式支持，用于字符串匹配、搜索和替换
import random
import math



def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:  # 如果存在第 i 个 GPU
        return torch.device(f'cuda:{i}')    # 返回第 i 个 GPU 设备
    return torch.device('cpu') # 若系统中无足够的GPU设备（即GPU数量<i+1），则返回CPU设备

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    # 如果存在可用的 GPU，则返回一个包含所有 GPU 设备的列表
    return devices if devices else [torch.device('cpu')]


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()] # 转换为tensor
    if resize: trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans) # compose整合步骤

    # Download data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",  # 数据集存储的位置
        train=True,  # 加载训练集（True则加载训练集）
        download=True,  # 如果数据集在指定目录中不存在，则下载（True才会下载）
        transform=trans,  # (使用什么格式转换,这里是对图片进行预处理，转换为tensor格式) 应用于图像的转换列表，例如转换为张量和归一化
        # transform=ToTensor(),  # (使用什么格式转换,这里是对图片进行预处理，转换为tensor格式) 应用于图像的转换列表，例如转换为张量和归一化
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,  # 加载测试集（False则加载测试集）
        download=True,
        transform=trans,
    )
    # 输出训练集和测试集的大小
    # print(f"\n训练集大小：{len(training_data)}, \n测试集的大小：{len(test_data)}")
    # print(f"索引到第一张图片，查看输入图像的通道数、高度和宽度：{training_data[0][0].shape}")

    # Create data loaders.
    # DataLoader()：batch_size每个批次的大小，shuffle=True则打乱数据
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader



'''（与 线性神经网络 的一样）
# 生成 “符合线性关系 y=Xw+b+噪声” 的合成数据集
# w: 权重向量（决定线性关系的斜率）
# b: 偏置项（决定线性关系的截距）
# num_examples: 要生成的样本数量
在指定正态分布中随机生成特征矩阵X，
然后根据传入的权重和偏置再加上随机生成的噪声计算得到标签向量y。
'''
def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    # 生成一个形状为 (num_examples, len(w)) 的矩阵,每个元素从均值为0、标准差为1的正态分布中随机采样
    X = torch.normal(0, 1, (num_examples, len(w)))
    print(f"X的形状{X.shape}")
    y = torch.matmul(X, w) + b  # 计算线性部分 Xw + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声（均值为0，标准差为0.01的正态分布）使数据更接近真实场景（避免完全线性可分）
    return X, y.reshape((-1, 1))  # 返回特征矩阵X和标签向量y, y.reshape((-1, 1)) 确保y是列向量（形状为 (num_examples, 1)）


# 封装了 Matplotlib 轴属性的常用设置
def set_axes(axes, xlabel=None, ylabel=None, xlim=None, ylim=None,
             xscale='linear', yscale='linear', legend=None):
    """设置绘图的轴属性"""
    if xlabel: axes.set_xlabel(xlabel)  # 设置x轴标签（如果提供）
    if ylabel: axes.set_ylabel(ylabel)  # 设置y轴标签（如果提供）
    if xlim: axes.set_xlim(xlim)        # 设置x轴范围（如 [0, 10]）（如果提供）
    if ylim: axes.set_ylim(ylim)        # 设置y轴范围（如 [0, 10]）（如果提供）
    axes.set_xscale(xscale)         # 设置x轴刻度类型（线性linear或对数log）
    axes.set_yscale(yscale)         # 设置y轴刻度类型（线性linear或对数log）
    if legend: axes.legend(legend)  # 添加图例文本列表（如 ['train', 'test']）（如果提供）
    axes.grid(True)                 # 显示背景网格线，提升可读性

# 绘图函数
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None,
         xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:', 'c-.', 'y-', 'k:'), figsize=(5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None: legend = [] # 默认图例为空列表（避免后续判断报错）
    # 创建画布（如果未提供外部axes）
    plt.figure(figsize=figsize)
    axes = axes if axes is not None else plt.gca()  # 获取当前轴

    # 如果X有一个轴，输出True。判断输入数据是否为一维（列表或一维数组）
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    # 标准化X和Y的形状：确保X和Y都是列表的列表（支持多条曲线）
    if has_one_axis(X):
        X = [X]  # 将一维X转换为二维（单条曲线）
    if Y is None: # 如果未提供Y，则X是Y的值，X轴为索引（如 plot(y)）
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y] # 将一维Y转换为二维
    if len(X) != len(Y): # 如果X和Y数量不匹配，复制X以匹配Y的数量
        X = X * len(Y)

    axes.clear() # 清空当前轴（避免重叠绘图）
    for x, y, fmt in zip(X, Y, fmts):
        if len(x): axes.plot(x, y, fmt) # 如果提供了x和y，绘制xy曲线
        else: axes.plot(y, fmt) # 如果未提供x，绘制y关于索引的曲线（如 plot(y)）

    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend) # 设置轴属性
    # 自动调整布局并显示图像
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域，防止标签溢出
    plt.show()

def plot_kernel_reg(x_test, y_truth, y_hat,
                    x_train, y_train):
    '''绘制所有的训练样本（样本由圆圈表示）
    不带噪声项的真实数据生成函数 f（标记为“Truth”）
    学习得到的预测函数（标记为“Pred”）
    y_hat   ：预测值数组
    x_train ：训练数据x坐标
    y_train ：训练数据y坐标
    x_test  ：测试数据x坐标(用于绘制真实曲线和预测曲线)
    y_truth ：真实值数组(对应x_test)
    '''
    # 使用 constrained_layout 替代 tight_layout
    plt.figure(figsize=(4, 3), constrained_layout=True) # 启用约束布局
    # 绘制  测试数据的真实输出 和 预测输出
    plt.plot(x_test, y_truth, label='Truth', linestyle='-', linewidth=1)
    plt.plot(x_test, y_hat, label='Pred', linestyle='--', linewidth=1, color='m')
    # plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])

    # 绘制 训练数据信息
    # plt.plot(x_train, y_train, 'o', alpha=0.5, label='Train Data')
    plt.scatter(x_train, y_train, color='orange', alpha=0.5, label='Train Data')

    # 设置图形属性
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 5)
    plt.ylim(-1, 5)
    plt.legend()
    plt.grid(True)
    plt.title('Kernel Regression Comparison')
    # plt.tight_layout() # 调整布局防止标签被截断
    plt.show()


def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


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


# 预测函数：生成prefix之后的新字符
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device) # 初始化隐藏状态，批量大小为1 (单序列预测)
    # 将prefix的第一个字符转换为索引
    # prefix[0]第一个字符，vocab[prefix[0]]获取第一个字符的索引
    outputs = [vocab[prefix[0]]]  # 存储生成的索引(用列表存储)
    # 辅助函数：获取当前输入 (形状为 (1, 1))
    # [outputs[-1]]获取索引列表中的最后一个，即 刚刚存进去的那个，也就是当前个
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 预热期：处理前缀中的剩余字符
    # 模型自我更新（例如，更新隐状态），但不进行预测
    for y in prefix[1:]: # 遍历前缀中除第一个字符外的所有字符
        _, state = net(get_input(), state)  # 前向传播(忽略输出，只更新隐藏状态)(把类当作函数使用，调用__call__)
        outputs.append(vocab[y])            # 将当前字符添加到输出列表
    # 预测阶段：生成新字符，预测num_preds步
    # 预热期结束后，隐状态的值比刚开始的初始值更适合预测，现在开始预测字符并输出
    for _ in range(num_preds):  # 预测指定数量的字符
        # y 形状为 (1, vocab_size)（批量大小=1）
        y, state = net(get_input(), state)          # 前向传播，获取预测输出
        # argmax(dim=1) 获取概率最高的词索引
        # int(y.argmax(dim=1).reshape(1)) 转换为 Python整数
        pred_idx = int(y.argmax(dim=1).reshape(1))  # 从输出中选择概率最高的索引
        outputs.append(pred_idx)                    # 将预测索引添加到输出列表
    # 将索引序列 转换回 字符序列
    # ''.join(...)将字符列表中的所有字符串（每个字符是一个长度为1的字符串）连接成一个字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    # len是查看矩阵的行数
    # y_hat.shape[1]就是取列数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 第2个维度为预测标签，取最大元素
        y_hat = y_hat.argmax(axis=1)  # 变成一列，列中每行元素为 行里的最大值下标

    # 将y_hat转换为y的数据类型然后作比较，cmp函数存储bool类型
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())  # 将正确预测的数量相加


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):  # 判断模型是否为深度学习模型
        net.eval()  # 将模型设置为评估模式

    # Accumulator(2)创建2个变量：正确预测的样本数总和、样本数
    metric = Accumulator(2)  # metric：度量，累加正确预测数、预测总数

    with torch.no_grad():  # 梯度不需要反向传播
        for X, y in data_iter:  # 每次从迭代器中拿出一个X和y
            # net(X)：X放在net模型中进行softmax操作
            # numel()函数：返回数组中元素的个数，在此可以求得样本数
            metric.add(accuracy(net(X), y), y.numel())

    # metric[0, 1]分别为网络预测正确数量和总预测数量
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 判断net模型是否为深度学习类型，将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()  # 要计算梯度，启用训练模式（启用Dropout/BatchNorm等训练专用层）

    # Accumulator(3)创建3个变量：训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)  # 用于跟踪训练损失、准确率和样本数
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)  # 前向传播：模型预测
        l = loss(y_hat, y)  # 计算损失（向量形式，每个样本一个损失值）

        # 判断updater是否为优化器
        if isinstance(updater, torch.optim.Optimizer):  # 使用PyTorch内置优化器
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()  # 把梯度设置为0（清除之前的梯度，避免梯度累加）
            l.mean().backward()  # 计算梯度（反向传播：计算梯度(对损失取平均)）l.mean()表示对批次损失取平均后再求梯度
            updater.step()  # 自更新（根据梯度更新模型参数）
        else:  # 使用自定义更新逻辑
            # 使用定制的优化器和损失函数
            # 自我实现的话，l出来是向量，先求和再求梯度
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失(平均损失)和训练精度，metric的值由Accumulator得到
    return metric[0] / metric[2], metric[1] / metric[2]


''' 梯度裁剪，目的：
防止梯度爆炸
稳定训练过程
特别适合 RNN 这类容易出现梯度问题的模型
'''
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    # 获取需要梯度的参数
    if isinstance(net, nn.Module): # PyTorch 模块：获取所有可训练参数
        params = [p for p in net.parameters() if p.requires_grad]
    else: # 自定义模型：使用模型自带的参数列表
        params = net.params
    # 计算所有参数的 梯度的 L2 范数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta: # 如果范数超过阈值θ，进行裁剪(将所有梯度按比例缩放)
        for param in params: # 保持梯度方向不变，只缩小幅度
            param.grad[:] *= theta / norm # 按比例缩放梯度

# 单迭代周期训练，以困惑度（Perplexity）作为评估指标
# 返回困惑度 和 训练速度(每秒处理的词元数量，用于衡量训练效率)
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, Timer() # 初始化状态和计时器
    metric = Accumulator(2)  # 训练损失之和,词元数量[loss_sum, token_count]
    for X, Y in train_iter: # 遍历数据批次
        # 状态初始化：如果是第一次迭代 或 使用随机抽样
        if state is None or use_random_iter:
            # 在第一次迭代 或 使用随机抽样时初始化state (创建全零的初始隐藏状态)
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 否则，分离状态，断开与历史计算图的连接，即 断开计算图（防止梯度传播到前一批次）
            # 训练循环中，对于非随机抽样（即顺序抽样），
            # 希望状态能跨批次传递，但又不希望梯度从当前批次反向传播到前一批次
            # (因为那样会导致计算图非常长，占用大量内存且可能梯度爆炸)。
            # 因此每次迭代开始时，需要将状态从计算图中分离出来
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 如果net是PyTorch模块，且状态非元组（例如GRU，它的状态是一个张量）
                # state对于nn.GRU是个张量
                state.detach_() # 用detach_()断开 状态与历史计算图 的连接(对张量进行原地分离操作)
            else:
                # 若状态是元组（例如LSTM的状态是两个张量，或者自定义的模型状态可能是元组）
                # state对于nn.LSTM 或 对于从零开始实现的模型 是个张量
                # LSTM 状态 或 自定义模型状态是元组
                for s in state:
                    s.detach_()
        # 准备数据：转置标签并展平
        # 先转置再展平 是为了 让标签的顺序与模型输出的顺序一致，从而正确计算损失
        # Y 原始形状: (batch_size, num_steps)
        # 转置后: (num_steps, batch_size); 展平后: (num_steps * batch_size)
        # 与 y_hat 形状 (num_steps * batch_size, vocab_size) 匹配
        y = Y.T.reshape(-1) # 形状: (num_steps * batch_size)
        X, y = X.to(device), y.to(device)   # 将数据转移到设备
        y_hat, state = net(X, state)        # 前向传播
        l = loss(y_hat, y.long()).mean()    # 计算每个词元的损失后取平均=对整个批次的加权平均
        # 反向传播
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()             # PyTorch 优化器
            l.backward()
            grad_clipping(net, 1)           # 梯度裁剪
            updater.step()
        else: # 自定义优化器
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1) # 因为损失已经取平均，batch_size=1
        metric.add(l * y.numel(), y.numel()) # 累积指标：损失 * 词元数，词元数
    perplexity = math.exp(metric[0] / metric[1]) # 计算困惑度 = exp(平均损失)
    speed = metric[1] / timer.stop() # 计算训练速度 = 词元数/秒
    return perplexity, speed


"""
# 评估函数
    定义精度评估函数：
    1、将数据集复制到显存中
    2、通过调用accuracy计算数据集的精度
"""
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):  # 判断net是否属于torch.nn.Module类(模型是否为深度学习模型)
        net.eval()  # 设置为评估模式（关闭Dropout和BatchNorm的随机性）
        if not device: # 如果没有指定设备，自动使用模型参数所在的设备（如GPU）
            device = next(iter(net.parameters())).device # 自动检测设备
    # 初始化计数器：累计 正确预测的数量 和 总预测的数量
    metric = Accumulator(2) # metric[0]=正确数, metric[1]=总数
    with torch.no_grad():  # 禁用梯度计算（加速评估并减少内存占用）
        for X, y in data_iter:  # 每次从迭代器中拿出一个X和y
            # 将数据X,y移动到指定设备（如GPU）
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # 计算预测值和准确率，并累加到metric中
            metric.add(accuracy(net(X), y), y.numel()) # 累加准确率和样本数
    # metric[0, 1]分别为网络预测正确数量和总预测数量
    return metric[0] / metric[1] # 计算准确率

"""
    定义GPU训练函数：
    1、为了使用gpu，首先需要将每一小批量数据移动到指定的设备（例如GPU）上；
    2、使用Xavier随机初始化模型参数；
    3、使用交叉熵损失函数和小批量随机梯度下降。
"""
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m): # 定义初始化参数，对线性层和卷积层生效
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) # Xavier初始化，保持输入输出的方差稳定
    net.apply(init_weights)  # 应用初始化到整个网络（初始化权重）

    # 在设备device上进行训练
    print('training on', device)
    net.to(device)  # 模型移至指定设备（如GPU）

    optimizer = torch.optim.SGD(net.parameters(), lr=lr) # 定义优化器：随机梯度下降（SGD），学习率为lr
    loss = nn.CrossEntropyLoss()  # 交叉熵损失
    # 初始化动画绘图器，用于动态绘制训练曲线
    animator = Animator(xlabel='epoch',
                               xlim=[1, num_epochs],
                               legend=['train loss', 'train acc', 'test acc'])
    # 初始化计时器和计算总批次数
    timer, num_batches = Timer(), len(train_iter) # 调用Timer函数统计时间
    # 开始训练循环
    for epoch in range(num_epochs):
        # Accumulator(3)创建3个变量：训练损失总和、训练准确度总和、样本数
        metric = Accumulator(3) # 用于跟踪训练损失、准确率和样本数
        net.train()  # 切换到训练模式（启用Dropout和BatchNorm的训练行为）
        for i, (X, y) in enumerate(train_iter):
            timer.start()           # 开始计时
            optimizer.zero_grad()   # 清空梯度
            X, y = X.to(device), y.to(device)   # 将数据移动到设备
            y_hat = net(X)          # 前向传播：模型预测
            l = loss(y_hat, y)      # 计算损失（向量形式，每个样本一个损失值）
            l.backward()            # 反向传播计算梯度
            optimizer.step()        # 更新参数
            with torch.no_grad(): # 禁用梯度计算后累计指标
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()            # 停止计时
            train_l = metric[0] / metric[2]     # 平均训练损失
            train_acc = metric[1] / metric[2]   # 平均训练准确率
            # 每训练完1/5的epoch 或 最后一个batch时，更新训练曲线
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        # 测试精度
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 测试集准确率
        animator.add(epoch + 1, (None, None, test_acc)) # 更新测试集准确率曲线
    print(f'最终结果：loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}') # 输出损失值、训练精度、测试精度
    print(f'训练速度（样本数/总时间）：{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}') # 设备的计算能力


''' 训练一个 字符级循环神经网络（RNN）模型
包含了训练循环、梯度裁剪、困惑度计算和文本生成预测
net             : 要训练的RNN模型（可以是PyTorch模块或自定义模型）
train_iter      : 训练数据迭代器
vocab           : 词汇表对象，用于索引和字符之间的转换
lr              : 学习率
num_epochs      : 训练的总轮数
device          : 训练设备（CPU或GPU）
use_random_iter : 是否使用随机采样（否则使用顺序分区）
'''
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    # 1. 初始化损失函数
    loss = nn.CrossEntropyLoss() # 使用交叉熵损失

    # 2. 初始化可视化工具：初始化动画器，用于绘制训练过程中的困惑度变化
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])

    # 3. 初始化优化器：根据net的类型选择不同的优化器
    if isinstance(net, nn.Module): # 如果是PyTorch模块，使用SGD优化器
        updater = torch.optim.SGD(net.parameters(), lr) # 则使用PyTorch的SGD优化器
    else:  # 如果是自定义模型，使用自定义的SGD优化器
        # 注意：这里的sgd函数需要三个参数：参数列表、学习率和批量大小（通过闭包捕获net.params和lr）
        # lambda batch_size: sgd(...) 创建闭包函数
        # 固定学习率 lr，动态传入 batch_size
        updater = lambda batch_size: sgd(net.params, lr, batch_size)

    # 4. 定义预测函数，用于生成 以给定前缀开头的文本（生成50个字符）
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device) # 设置预测函数

    # 5. 主训练循环：训练和预测
    for epoch in range(num_epochs):
        # 5.1 训练一个epoch，返回困惑度（ppl）和训练速度（speed）（每秒处理多少个词元）
        # speed 表示每秒处理的词元数量，用于衡量训练效率
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        # 5.2 每10个epoch进行一次评估和可视化
        if (epoch + 1) % 10 == 0: # 每10个epoch
            print(predict('time traveller'))  # 使用前缀'time traveller'生成文本并打印
            animator.add(epoch + 1, [ppl]) # 将当前epoch的困惑度添加到动画中

    # 6. 训练结束后的最终评估：打印最终的困惑度和速度
    # 困惑度：语言模型质量指标（越低越好）
    # 处理速度：每秒处理的词元数量
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    # 使用两个不同的前缀生成文本
    print(predict('time traveller'))
    print(predict('traveller'))

"""
在序列中屏蔽不相关的项（将超出有效长度的位置设置为指定值）
X: 输入张量，可以是二维(batch_size, seq_len)或三维(batch_size, seq_len, features)
valid_len: 每个序列的有效长度，形状为(batch_size,)
value: 用于填充无效位置的值，默认为0
返回: 掩码后的张量，形状与X相同
"""
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项
    在序列中屏蔽不相关的项（将超出有效长度的位置设置为指定值）
    X: 输入张量，可以是二维(batch_size, seq_len)或三维(batch_size, seq_len, features)
    valid_len: 每个序列的有效长度，形状为(batch_size,)
    value: 用于填充无效位置的值，默认为0
    返回: 掩码后的张量，形状与X相同
    """
    maxlen = X.size(1) # 获取序列的最大长度  .size(1)即第2维

    # 创建位置索引 [0, 1, 2, ..., maxlen-1]，使用与X相同的设备（CPU/GPU）
    arange = torch.arange(maxlen, dtype=torch.float32, device=X.device)

    # [None, :] 将arange   扩展为 行向量 (1, maxlen)，代表当前第几列，即 该行的第几个长度
    # [:, None] 将valid_len扩展为 列向量 (batch_size,1)，代表当前行有几个有效长度
    # 广播比较：每个序列的位置索引 < 该序列的有效长度
    mask = arange[None, :] < valid_len[:, None] # 创建掩码矩阵
    """ 即 行向量arange_row < 列向量valid_len_col
    [[0<1, 1<1, 2<1]  → [True, False, False]
     [0<2, 1<2, 2<2]] → [True, True, False]
     注：假设 
     行向量 arange_row 为 [[0, 1, 2]]
     列向量 valid_len  为 [[1], [2]]
    """
    X[~mask] = value # 将掩码取反，并将无效位置设置为指定值
    return X

# 带遮蔽的softmax交叉熵损失函数类
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """ 带遮蔽的softmax交叉熵损失函数
    pred : 模型预测值，形状 (batch_size,num_steps,vocab_size)
    label: 真实标签，形状 (batch_size,num_steps)
    valid_len: 每个序列的有效长度，形状 (batch_size,)
    返回: 加权后的损失，形状 (batch_size,)
    """
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label) # 与label形状相同的权重矩阵，初始值为1
        weights = sequence_mask(weights, valid_len) # 使用序列掩码函数，将超出有效长度的位置权重设为0
        self.reduction='none' # 设置父类CrossEntropyLoss为不自动求平均（返回每个位置的损失）
        # 调用父类计算未加权的交叉熵损失
        # 注：CrossEntropyLoss期望pred的形状为 (batch_size, num_classes, num_steps)
        # 所以使用permute(0, 2, 1)调整维度顺序
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), # 调整为 (batch_size, vocab_size, num_steps)
            label)                 # 形状 (batch_size, num_steps)
        # unweighted_loss * weights 应用权重：有效位置保留原损失值，无效位置乘以0
        # .mean(dim=1) 对每个序列在时间步维度求平均
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss # 返回: 加权后的损失，形状 (batch_size,)

# 训练序列到序列模型
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """ 训练序列到序列模型
    net: 编码器-解码器模型
    data_iter: 数据迭代器，提供训练批次
    lr: 学习率
    num_epochs: 训练轮数
    tgt_vocab: 目标语言词表
    device: 计算设备(CPU/GPU)
    """
    def xavier_init_weights(m):
        '''权重初始化函数（Xavier初始化）'''
        # nn.init.xavier_uniform_：根据输入/输出维度自适应缩放权重，适合Sigmoid/Tanh激活函数
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight) # 线性层权重初始化
        if type(m) == nn.GRU:
            for param in m._flat_weights_names: # 遍历GRU的所有参数名
                if "weight" in param: # 对权重参数进行Xavier初始化
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)  # 应用初始化
    net.to(device)                  # 移动模型到指定设备
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # Adam优化器
    loss = MaskedSoftmaxCELoss() # 掩码交叉熵损失（忽略填充词元）
    net.train() # 设置为训练模式
    animator = Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs]) # 可视化工具
    for epoch in range(num_epochs): # 每轮迭代
        timer = Timer() # 计时器实例化
        metric = Accumulator(2)  # 累计 训练损失总和，有效词元数量
        for batch in data_iter: # 当前批量数据(批次训练)
            optimizer.zero_grad() # 梯度清零(重置)
            # 获取批量数据并移动到设备
            # 源语言序列，源语言有效长度；目标语言序列，目标语言有效长度
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # 强制教学：构造解码器输入（Teacher Forcing）
            #[tgt_vocab['<bos>']] * Y.shape[0]生成长度为Y.shape[0]
            # （即当前批次的句子数量，batch_size）的列表，元素均为 <bos> 的索引
            # .reshape(-1, 1)将张量形状 (batch_size,)→(batch_size,1)即每行一个<bos>，方便后续拼接
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1) # 起始符号
            # 将起始符号与(目标序列每一行去除掉最后一列数据后的结果数据)沿轴2即列拼接，即增加列数
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 拼接<bos>和去掉最后一个词的目标序列

            # 强制教学 vs 自回归：
            # 代码中显式使用真实标签构造dec_input，属于【强制教学】
            # 若需自回归训练，需修改dec_input为模型预测结果（通常结合调度采样）
            Y_hat, _ = net(X, dec_input, X_valid_len) # 前向传播（编码器+解码器）模型预测

            l = loss(Y_hat, Y, Y_valid_len) # 计算损失（仅对有效词元）
            l.sum().backward()              # 损失函数的标量进行“反向传播”
            grad_clipping(net, 1)     # 梯度裁剪防止爆炸(梯度裁剪阈值为1)(裁剪梯度必须在反向传播后)

            # 更新指标
            num_tokens = Y_valid_len.sum()  # 当前批次的有效词元总数
            optimizer.step()                # 根据梯度更新模型参数
            with torch.no_grad():           # 避免在更新指标时产生不必要的计算图
                metric.add(l.sum(), num_tokens) # 累加 总损失和词元数
        if (epoch + 1) % 10 == 0: # 每10轮可视化训练损失
            avg_loss = metric[0] / metric[1]  # 平均损失
            animator.add(epoch + 1, (avg_loss,))
    # 最终训练结果
    avg_loss = metric[0] / metric[1]
    tokens_per_sec = metric[1] / timer.stop()
    print(f'loss {avg_loss:.3f}, {tokens_per_sec:.1f}  tokens/sec on {str(device)}')


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测
    net         : 训练好的序列到序列模型
    src_sentence: 源语言句子（字符串）
    src_vocab   : 源语言词汇表
    tgt_vocab   : 目标语言词汇表
    num_steps   : 序列的最大长度（包括填充）
    device      : 使用的设备（如'cpu'或'cuda'）
    save_attention_weights: 是否保存注意力权重（默认为False）
    """
    net.eval() # 预测时将net设为评估模式
    # 1. 处理源语言输入序列
    # 将源句子统一转小写 → 按空格分词 → 从词转为词索引 → 末尾加上结束符<eos>
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    # 创建有效长度张量（实际非填充token数）
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 对源序列进行填充/截断，使其长度等于num_steps
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴（batch_size=1）并转换为张量
    # 在第0维添加一个维度，即 批量大小维度
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # 2. 编码器处理（长度可变序列 → 固定形状的编码状态）
    # 通过编码器获取上下文信息（enc_outputs包含隐藏状态/记忆单元）
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 3. 初始化解码器状态
    # 使用编码器输出初始化解码器的初始状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 4. 准备解码器初始输入
    # 创建以<bos>（序列开始符）开头的目标序列张量，添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], [] # 初始化输出序列和注意力权重容器
    # 5. 自回归生成序列
    for _ in range(num_steps): # 循环生成最多num_steps个token
        Y, dec_state = net.decoder(dec_X, dec_state) # 解码器前向传播（生成 预测和更新状态）
        # 使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2) # 选择预测概率最高的词元（贪婪搜索策略）第2维的最大值索引
        # 提取预测词元的索引（移除批量维度→类型转为32位整数→从单元素张量中提取标量值）
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights: # 可选：保存当前时间步的注意力权重
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']: # 终止条件：预测到结束符<eos>
            break
        output_seq.append(pred) # 将预测词元添加到输出序列
    # 6. 后处理并返回结果
    # 将索引序列 转换为 目标语言词元字符串
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU：评估 预测序列与真实标签序列的相似度"""
    # 预测序列与标签序列皆以空格分割为token词元列表
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens) # 获取序列长度
    # 计算简洁惩罚（Brevity Penalty, BP）
    # 当预测长度 >  参考长度时：BP=1 即无惩罚（这里用min(0, ...)来调整）任何数的0次方=1
    # 当预测长度 <= 参考长度时：BP = e^(1 - len_ref/len_pred)，即 使用指数形式惩罚过短序列
    BP = math.exp(min(0, 1 - len_label / len_pred))
    score = BP # 初始化BLEU分数（先乘以BP）
    for n in range(1, k + 1): # 遍历1-gram到k-gram，计算加权精度
        # 统计标签序列中的n-gram
        label_subs = collections.defaultdict(int) # 存储标签序列中n-gram的出现次数(存入字典)
        for i in range(len_label - n + 1):
            ngram = ' '.join(label_tokens[i: i + n]) # 创建n-gram字符串(用空格连接词元)
            label_subs[ngram] += 1 # 记录出现次数

        # 在预测序列中匹配n-gram
        num_matches = 0 # 匹配的n-gram计数
        for i in range(len_pred - n + 1):
            ngram = ' '.join(pred_tokens[i: i + n]) # 预测序列中的待匹配成员
            if label_subs[ngram] > 0: # 若n-gram存在于参考译文中且计数>0
                num_matches += 1
                label_subs[ngram] -= 1 # 避免重复匹配，且截断计数(预测中<=标签中的出现次数)
        # 计算当前n-gram的精度（p_n）
        # 当 len_pred < n 时，p_n 设为 0（无法提取 n-gram）
        p_n = num_matches / (len_pred - n + 1) if len_pred >= n else 0 # 避免除零错误
        # 累积到总分数，即 累积加权精度
        # （原代码问题：权重应为固定值如0.25，而非0.5^n）
        weight = 1 / k  # BLEU通常采用均匀权重
        weight = math.pow(0.5, n) # 使用指数加权：权重=0.5^n
        score *= math.pow(p_n, weight) # 几何平均
    return score


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """ 显示矩阵热图
    matrices: 4D数组 (要显示的行数，要显示的列数，查询的数目，键的数目)
             可以是PyTorch张量或NumPy数组
    xlabel (str)    : x轴标签
    ylabel (str)    : y轴标签
    titles (list)   : 子图标题列表（可选，数量应等于要显示的列数）
    figsize (tuple) : 每个子图的图形大小（英寸）
    cmap (str)      : 颜色映射名称（如'Reds', 'viridis'）
    SVG(Scalable Vector Graphics，可缩放矢量图形)适合简单图形(图标、图表、Logo),基于XML的矢量图形格式
    """
    # 设置Matplotlib使用SVG后端（在Jupyter或PyCharm中生效）
    plt.rcParams['figure.dpi'] = 100            # 可选：调整分辨率
    plt.rcParams['svg.fonttype'] = 'none'       # 确保文本可编辑（SVG 特性）
    plt.rcParams['figure.facecolor'] = 'white'  # 可选：设置背景色

    # 获取矩阵的行数和列数
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]

    # 创建子图
    fig, axes = plt.subplots(num_rows, num_cols,       # 子图的行数和列数
                             figsize=figsize,          # 图形大小（宽, 高）
                             sharex=True, sharey=True, # 所有子图 共享xy轴（避免重复标签）
                             squeeze=False) # squeeze=False 强制axes始终是二维，即使只有一行或一列
    pcm = None # 初始化一个pcm变量用于颜色条

    # 遍历所有子图
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]         # 获取当前子图
            matrix = matrices[i, j] # 获取当前子图对应的矩阵 (查询数×键数)

            # 处理不同类型的输入
            if hasattr(matrix, 'detach'):  # 如果是PyTorch张量
                matrix_data = matrix.detach().numpy()
            elif isinstance(matrix, np.ndarray):  # 如果是NumPy数组
                matrix_data = matrix
            else:  # 其他类型尝试转换为NumPy数组
                matrix_data = np.array(matrix)

            pcm = ax.imshow(matrix_data, cmap=cmap) # 绘制热图，cmap为颜色映射
            # 设置标签（只在边缘子图显示）
            if i == num_rows - 1:
                ax.set_xlabel(xlabel) # Keys
            if j == 0:
                ax.set_ylabel(ylabel) # Queries

            # 设置标题（使用titles[j]确保每列标题一致）
            if titles and j < len(titles):
                ax.set_title(titles[j])

    plt.tight_layout() # 调整布局防止重叠
    # 添加全局颜色条（使用最后一个子图的pcm）
    if pcm is not None:
        # shrink=0.6 颜色条的长度比例缩放，缩小为60%
        # location='right' 颜色条位置处右侧
        fig.colorbar(pcm, ax=axes, shrink=0.6, location='right')
    plt.show()

def masked_softmax(X, valid_lens):
    """带掩码的softmax函数，用于处理变长序列：通过在最后一个轴上掩蔽元素来执行softmax操作
    掩膜后，则只保留了每个序列中有效的特征维度（无效的都变为0）
    X         : 3D张量，待掩膜的矩阵，
                形状为 (批次batch_size, 序列长度seq_length, 特征维度feature_dim)
    valid_lens: 每个样本的有效长度（非填充位置），1D或2D张量
                - None: 不使用掩码
                - 1D: 每个序列的有效长度
                - 2D: 每个序列中每个位置的有效长度
    返回: 经过掩蔽处理的softmax结果，形状与X相同
    """
    if valid_lens is None: # 没有提供有效长度，则直接执行标准softmax
        return nn.functional.softmax(X, dim=-1) # dim=-1是沿最后一维进行softmax归一化
    else:
        shape = X.shape # 张量形状
        if valid_lens.dim() == 1:
            # 当valid_lens是1D时：扩展为与X第二维匹配
            # 如 batch_size=2, seq_length=3，valid_lens=[2,1] (两个序列的有效长度分别为2和1)
            # -> 扩展为 [len1, len1, len1, len2, len2, len2]，即 [2,2,2,1,1,1]
            # 输入的有效长度为1维，每个元素对应原矩阵中每个序列(每行)的有效长度
            # 因此将有效长度向量中，每个元素重复，重复次数对应每批次的序列数量(原矩阵的第1维)
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 当valid_lens是2D时：展平为一维张量
            # 因为有效长度为2d时，就是已经细分了每个序列自己的有效长度，因此直接展平即可
            # 例如valid_lens = [[2, 2, 1], [1, 0, 0]](每个序列中每个位置的有效长度)
            # - 展平后: [2, 2, 1, 1, 0, 0]
            valid_lens = valid_lens.reshape(-1)

        # 核心掩蔽操作：
        # 1. X.reshape(-1, shape[-1])将X重塑为二维张量(batch_size * seq_length, feature_dim)
        # 2. 使用sequence_mask生成掩码：
        #    - 有效位置保持原值
        #    - 无效位置（超出有效长度的位置）被替换为 -1e6（极大负值）
        # 3. 经过softmax后，无效位置输出概率接近0
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        # 恢复原始形状并执行softmax
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# 计时器
class Timer:  # @save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


# 实用程序类，示例中创建两个变量：正确预测的数量 和 预测总数
class Accumulator:  # @save
    """在n个变量上累加"""
    def __init__(self, n): # 初始化根据传进来n的大小来创建n个空间，全部初始化为0.0
        self.data = [0.0] * n

    # 把原来类中对应位置的data和新传入的args做a + float(b)加法操作然后重新赋给该位置的data，从而达到累加器的累加效果
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    # 重新设置空间大小并初始化。
    def reset(self):
        self.data = [0.0] * len(self.data)

    # 实现类似数组的取操作
    def __getitem__(self, idx):
        return self.data[idx]


# 实用程序类，动画绘制器，动态绘制数据
class Animator:  # @save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        # 创建图形和坐标轴
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]  # 确保axes是列表形式（即使只有1个子图）
        # 设置坐标轴配置的函数
        def set_axes(ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            if legend:
                ax.legend(legend)
            ax.grid()
        # 使用lambda函数捕获参数
        self.config_axes = lambda: [set_axes(ax, xlabel, ylabel, xlim, ylim,
                                             xscale, yscale, legend) for ax in self.axes]
        self.X, self.Y, self.fmts = None, None, fmts
        plt.tight_layout()  # 自动调整布局防止标签重叠
        plt.ion()  # 开启交互模式，使图形可以实时更新

    def add(self, x, y):
        # 向图表中添加多个数据点
        # x: x值或x值列表
        # y: y值或y值列表
        # hasattr(y, "__len__")：检查 y 是否为多值（如列表或数组）
        if not hasattr(y, "__len__"):
            y = [y]  # 如果y不是列表/数组，转换为单元素列表
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n  # 如果x是标量，扩展为与y长度相同的列表
        if not self.X:
            self.X = [[] for _ in range(n)]  # 初始化n条曲线的x数据存储
        if not self.Y:
            self.Y = [[] for _ in range(n)]  # 初始化n条曲线的y数据存储
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)  # 添加x数据
                self.Y[i].append(b)  # 添加y数据
        for ax in self.axes: # 清除并重新绘制所有子图
            ax.cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            for ax in self.axes:
                ax.plot(x, y, fmt) # 重新绘制所有曲线

        self.config_axes()
        self.fig.canvas.draw()  # 更新画布
        self.fig.canvas.flush_events()  # 刷新事件
        time.sleep(0.1)  # 添加短暂延迟以模拟动画效果
        plt.show() # pycharm社区版没有科学模块，通过在循环里show来实现动画效果

    def close(self):
        """关闭图形"""
        plt.ioff()  # 关闭交互模式
        plt.close(self.fig)


# 下载器类：下载和缓存数据集
class C_Downloader:
    def __init__(self, data_url = 'http://d2l-data.s3-accelerate.amazonaws.com/'):
        # DATA_HUB字典，将数据集名称的字符串映射到数据集相关的二元组上
        # DATA_HUB为二元组：包含数据集的url和验证文件完整性的sha-1密钥
        self.DATA_HUB = dict()
        self.DATA_URL = data_url # 数据集托管在地址为DATA_URL的站点上

    ''' download
    下载数据集，将数据集缓存在本地目录（默认为../data）中，并返回下载文件的名称
    若缓存目录中已存在此数据集文件，且其sha-1与存储在DATA_HUB中的相匹配，则使用缓存的文件，以避免重复的下载
    name：要下载的文件名，必须在DATA_HUB中存在
    cache_dir：缓存目录，默认为../data
    sha-1：安全散列算法1
    '''
    def download(self, name, cache_dir=os.path.join('..', 'data')):  # @save
        """下载一个DATA_HUB中的文件，返回本地文件名"""
        # 检查指定的文件名是否存在于DATA_HUB中
        # 如果不存在，则抛出断言错误，提示用户该文件不存在
        # 断言检查：确保name在DATA_HUB中存在，避免下载不存在的文件
        assert name in self.DATA_HUB, f"{name} 不存在于 {self.DATA_HUB}"
        url, sha1_hash = self.DATA_HUB[name] # 从DATA_HUB中获取该文件的URL和SHA-1哈希值

        # 若目录不存在，则创建目录
        # exist_ok=True：若目录已存在，也不会抛出错误
        os.makedirs(cache_dir, exist_ok=True)

        # 构建本地文件路径
        # 从URL中提取文件名（通过分割URL字符串获取最后一个部分）
        # 并将该文件名与缓存目录组合成完整的本地文件路径
        fname = os.path.join(cache_dir, url.split('/')[-1])

        if os.path.exists(fname): # 检查本地文件是否已存在
            sha1 = hashlib.sha1() # 计算本地文件的SHA-1哈希值(shal.sha1()：创建一个字符串hashlib_，并将其加密后传入)
            with open(fname, 'rb') as f:
                # 读取文件内容，每次读取1MB的数据块，以避免大文件占用过多内存
                while True:
                    data = f.read(1048576) # 1048576 bytes = 1MB
                    if not data:
                        break
                    sha1.update(data) # 更新哈希值

            # 比较计算出的哈希值与DATA_HUB中存储的哈希值
            if sha1.hexdigest() == sha1_hash:
                # 若哈希值匹配，说明文件完整且未被篡改，直接返回本地文件路径（命中缓存）
                return fname  # 命中缓存

        # 如果本地文件不存在或哈希值不匹配，则从URL下载文件
        print(f'正在从{url}下载{fname}...')

        # 使用requests库发起HTTP GET请求，stream=True表示以流的方式下载大文件
        # verify=True表示验证SSL证书（确保下载的安全性）
        r = requests.get(url, stream=True, verify=True)

        # 将下载的内容写入到本地文件中
        with open(fname, 'wb') as f:
            f.write(r.content) # 将请求的内容写入文件
        return fname # 返回本地文件路径

    ''' 
    下载并解压一个zip或tar文件
    name：要下载并解压的文件名，必须在DATA_HUB中存在
    folder：解压后的目标文件夹名（可选）
    '''
    def download_extract(self, name, folder=None):  # @save
        """下载并解压zip/tar文件"""
        fname = self.download(name) # 调用download函数下载指定的文件，获取本地文件路径
        base_dir = os.path.dirname(fname) # 获取缓存目录路径（即下载文件所在的目录）
        data_dir, ext = os.path.splitext(fname) # 分离文件名和扩展名

        if ext == '.zip':               # 如果是zip文件，使用zipfile.ZipFile 打开文件
            fp = zipfile.ZipFile(fname, 'r')
        elif ext in ('.tar', '.gz'):    # 如果是tar或gz文件，使用tarfile.open 打开文件
            fp = tarfile.open(fname, 'r')
        else:                           # 如果文件扩展名不是zip、tar或gz，抛出断言错误
            assert False, '只有zip/tar文件可以被解压缩'

        fp.extractall(base_dir) # 将文件解压到缓存目录中
        # 返回解压后的路径
        # 如果指定了folder参数，返回解压后的目标文件夹路径
        # 否则返回解压后的文件路径（即去掉扩展名的文件名）
        return os.path.join(base_dir, folder) if folder else data_dir

    # 将使用的所有数据集从DATA_HUB下载到缓存目录中
    def download_all(self):  # @save
        """下载DATA_HUB中的所有文件"""
        for name in self.DATA_HUB:
            self.download(name)


'''
Vocab类：构建文本词表（Vocabulary），管理词元与索引的映射关系
功能：
构建词表，管理词元与索引的映射关系，支持：
词元 → 索引（token_to_idx）
索引 → 词元（idx_to_token）
过滤低频词
保留特殊词元（如 <unk>未知, <pad>填充符, <bos>起始符, <eos>结束符）
'''
class Vocab:  #@save
    """文本词表"""
    # tokens：原始词元列表（一维或二维）
    # min_freq：最低词频阈值，低于此值的词会被过滤
    # reserved_tokens：预定义的特殊词元（如 ["<pad>", "<bos>"]）
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None: tokens = []
        if reserved_tokens is None: reserved_tokens = []

        # 统计词频，按出现频率排序
        counter = count_corpus(tokens) # 统计词频
        # key=lambda x: x[1] 指定排序依据为第二个元素
        # reverse=True 降序排序
        # _var：弱私有(Protected),
        # 表示变量是内部使用的，提示开“不要从类外部直接访问”，但实际上仍然可访问(Python不会强制限制)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True) # 词元频率(词频)，按频率降序排序
        # 初始化词表，未知词元的索引为0（<unk>）
        # idx_to_token：索引 → 词元，索引0 默认是 <unk>（未知词元），后面是reserved_tokens
        # token_to_idx：词元 → 索引，是 idx_to_token 的反向映射
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)} # 字典
        # 按词频从高到低添加词元，过滤低频词
        for token, freq in self._token_freqs:
            if freq < min_freq: # 跳过低频词
                break
            if token not in self.token_to_idx: # 若词元不在token_to_idx中，则添加到词表
                self.idx_to_token.append(token) # 压入新词元
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # 新词元对应的索引

    # __len__用于定义对象的长度行为。
    # 对类的实例调用len()时，Python会自动调用该实例的__len__方法
    def __len__(self): # 词表大小（包括 <unk> 和 reserved_tokens）
        return len(self.idx_to_token) # 返回词表大小

    # 词表索引查询：支持单个词元或词元列表查询 ↓
    # vocab["hello"] → 返回索引（如 1）
    # vocab[["hello", "world"]] → 返回索引列表 [1, 2]
    # 若词元不存在，返回 unk 的索引（默认 0）
    # __getitem__定义对象如何响应obj[key]这样的索引操作，实现后 实例可像列表或字典一样通过方括号[]访问元素
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)): # 若传入参数不为列表或元组，而是单个
            # 字典的内置方法 dict.get(key, default) 用于安全地获取字典中某个键（key）对应的值
            # 若键不存在，则返回指定的默认值（default），而不会抛出 KeyError 异常
            return self.token_to_idx.get(tokens, self.unk)   # 单个词元返回索引，未知词返回0
        return [self.__getitem__(token) for token in tokens] # 递归处理列表

    # 索引转词元
    # 支持单个索引或索引列表转换：
    # vocab.to_tokens(1) → 返回词元（如 "hello"）
    # vocab.to_tokens([1, 2]) → 返回词元列表 ["hello", "world"]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)): # 若传入参数不为列表或元组，而是单个
            return self.idx_to_token[indices]  # 单个索引返回词元
        return [self.idx_to_token[index] for index in indices] # 递归处理列表

    @property
    def unk(self):  # 返回未知词元的索引（默认为0）
        return 0

    @property
    def token_freqs(self): # 返回原始词频统计结果（降序排列）
        return self._token_freqs # 返回词频统计结果

# 辅助函数：统计词元（tokens）的频率，返回一个 Counter对象
def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    # 如果tokens是空列表或二维列表（如句子列表），则展平为一维列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    # collections.Counter统计每个词元的出现次数，返回类似{"hello":2, "world":1}的字典
    return collections.Counter(tokens)



'''
加载文本数据
# 下载器与数据集配置
# 为 time_machine 数据集注册下载信息，包括文件路径和校验哈希值（用于验证文件完整性）
downloader = common.C_Downloader()
DATA_HUB = downloader.DATA_HUB  # 字典，存储数据集名称与下载信息
DATA_URL = downloader.DATA_URL  # 基础URL，指向数据集的存储位置
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
'''
def read_time_machine(downloader):  #@save
    """将时间机器数据集加载到文本行的列表中"""
    # 通过 downloader.download('time_machine') 获取文件路径
    with open(downloader.download('time_machine'), 'r') as f:
        lines = f.readlines() # 逐行读取文本文件
    # 用正则表达式 [^A-Za-z]+ 替换所有非字母字符为空格
    # 调用 strip() 去除首尾空格，lower() 转换为小写
    # 返回值：处理后的文本行列表（每行是纯字母组成的字符串）
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# 读取“英语－法语”数据集(神经网络机器翻译nmt中要用到的)
# downloader将下载器类对象传入，以调用“下载并解压”的功能
def read_data_nmt(downloader):
    """载入“英语－法语”数据集"""
    # 1. 下载并解压数据集（若本地不存在）
    # 自动从DATA_HUB下载压缩包并解压到本地缓存目录
    data_dir = downloader.download_extract('fra-eng')
    # 2. 打开解压后的法语文本文件（fra.txt）
    #    - 文件路径格式：{解压目录}/fra.txt
    #    - 使用UTF-8编码读取（支持法语特殊字符如é, ç等）
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read() # 3. 返回整个文件内容作为字符串

'''
预处理文本数据
(统一空格、统一小写、单词与符号见空格隔开)
原始文本: "Ça va?\u202fOui!"
处理步骤:
1. 替换特殊空格: "Ça va? Oui!"
2. 转为小写: "ça va? oui!"
3. 标点前加空格: "ça va ?  oui !"
最终输出: "ça va ?  oui !"
'''
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char): # 判断是否需要在标点符号前添加空格
        """检查当前字符是否是标点且前一个字符不是空格"""
        punctuations = set(',.!?') # 需要处理的标点符号集合
        # 满足两个条件：1.当前字符是标点 2.前一个字符不是空格
        return char in punctuations and prev_char != ' '

    # 使用空格替换不间断空格(统一空白字符)
    # \u202f：法语中常用的不间断窄空格(narrow no-break space)
    # \xa0：不间断空格(no-break space)
    # 皆统一替换为普通空格
    # 使用小写字母替换大写字母
    # .lower()将所有字母转为小写（法语有重音符号的小写形式）
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在单词和标点符号之间插入空格
    # 例如："Hello,world" -> "Hello , world"
    out = [
           # 如果当前字符是标点且前一个字符不是空格，则在该标点前插入空格
           ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           # 遍历文本中的每个字符及其索引
           for i, char in enumerate(text)]
    return ''.join(out) # 将处理后的字符列表重新组合成字符串

# 词元化函数：支持按单词或字符拆分文本
# lines：预处理后的文本行列表
# token：词元类型，可选 'word'（默认）或 'char
# 返回值：嵌套列表，每行对应一个词元列表
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]  # 按空格分词
    elif token == 'char':
        return [list(line) for line in lines]   # 按字符拆分
    else:
        print('错误：未知词元类型：' + token)


"""
词元化“英语－法语”数据集
text: 原始文本数据（预处理后的字符串）
num_examples: 可选参数，限制处理的样本数量
返回:
source: 词元化后的 英语 句子列表（每个句子是词元列表）
target: 词元化后的 法语 句子列表（每个句子是词元列表）
"""
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], [] # 初始化存储词元化结果的列表
    # 按行分割文本（每行应为"英语句子\t法语句子"的格式）
    for i, line in enumerate(text.split('\n')):
        # 若指定了样本数量限制，且已超过限制，则终止循环
        if num_examples and i > num_examples:
            break
        parts = line.split('\t') # 按制表符分割每行，得到英语和法语句子
        if len(parts) == 2: # 确保分割后确实有两部分（有效数据行）
            # 对英语句子进行分词（按空格分割）
            # 例如："hello world" → ["hello", "world"]
            # 例如："go ." → ["go", "."]
            source.append(parts[0].split(' '))
            # 对法语句子进行分词（按空格分割）
            # 注意：这里假设法语句子已经过preprocess_nmt处理，标点与单词间有空格
            # 例如："ça alors !" → ["ça", "alors", "!"]
            target.append(parts[1].split(' '))
    return source, target # 返回分词后的双语平行语料


""" 绘制两个列表长度分布的对比直方图
legend (list): 图例标签，例如 ['Source', 'Target']
xlabel  (str): x轴标签，例如 'Sentence Length'
ylabel  (str): y轴标签，例如 'Count'
xlist  (list): 第一个列表（如源语言句子列表）
ylist  (list): 第二个列表（如目标语言句子列表）
"""
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    plt.figure(figsize=(6, 4)) # 设置图形大小

    # 计算两个列表的长度分布
    # 对xyList中每一个元素都计算len(l)，然后组成新列表
    x_lengths = [len(l) for l in xlist]  # 第一个列表的长度序列
    y_lengths = [len(l) for l in ylist]  # 第二个列表的长度序列
    # 绘制直方图，返回补丁对象用于后续修改
    _, _, patches = plt.hist(
        [x_lengths, y_lengths],
        # bins='auto', # bins='auto' 让matplotlib自动选择合适的柱子数量
        # alpha=0.5,   # alpha=0.5 设置透明度以便观察重叠部分
        # label=legend
    )
    plt.xlabel(xlabel) # 设置坐标x轴标签
    plt.ylabel(ylabel) # 设置坐标y轴标签
    # 为第二个直方图添加斜线填充图案（增强视觉区分）
    # patches[1]对应第二个输入的数据（ylist）
    for patch in patches[1].patches:
        patch.set_hatch('/') # 设置填充样式为斜线
    plt.legend(legend) # 显示图例
    plt.tight_layout() # 自动调整布局防止标签重叠
    plt.show()

''' 截断或填充文本序列到固定长度
line         : 输入的文本序列（通常是词元ID列表，如 [1,5,23,4]）
num_steps    : 目标固定长度（时间步数/词元数量）
padding_token: 用于填充的特殊词元（如 '<pad>' 对应的ID）
返回值：处理后的序列（截断或填充后的固定长度序列）
'''
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps: # 若序列长度超限，则直接截断
        return line[:num_steps]  # 截断
    # 若序列长度不足，则直接填充至目标长度
    return line + [padding_token] * (num_steps - len(line)) # 填充


""" 将机器翻译的文本序列转换成小批量
lines    : List[List[str]]，文本序列列表（每个序列是单词列表，如[["I","love"], ["Halo","world"]]）
vocab    : common.Vocab，词表对象，用于将单词映射为整数ID
num_steps: int，序列的最大长度（截断或填充的目标长度）
返回:
array    : torch.Tensor，形状为(batch_size, num_steps)，包含填充后的序列ID
valid_len: torch.Tensor，形状为(batch_size,)，表示每个序列的有效长度（不含<pad>）
"""
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    # 输入lines: [["I", "love"], ["Hello", "world"]]
    # 输出lines: [[vocab["I"], vocab["love"]], [vocab["Hello"], vocab["world"]]]
    lines = [vocab[l] for l in lines]             # 将每个单词转换为词表中的整数ID
    # 输出lines: [[vocab["I"], vocab["love"], vocab["<eos>"]],
    #             [vocab["Hello"], vocab["world"], vocab["<eos>"]]]
    lines = [l + [vocab['<eos>']] for l in lines] # 在每个序列末尾添加<eos>（句子结束符）

    # 对每个序列进行截断或填充，使其长度为num_steps
    # 此时array的形状将会是(batch_size, num_steps)，batch_size即为lines的行数，即句子个数
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])

    # 计算每个序列的有效长度（非<pad>的token数量）
    # 布尔掩码 (array != vocab['<pad>']) 标记非填充位置
    # 将张量中不等于<pad> ID的位置标记为1，求和得到有效长度
    # .sum(1) 沿列求和降维，即列数清0
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


""" 返回翻译数据集的迭代器和词表
downloader: 数据下载器（如d2l.Downloader） - 用于获取原始数据
batch_size: int - 每个批量的样本数
num_steps: int - 序列的最大长度（截断或填充的目标长度）
num_examples: int - 使用的样本数量（默认600，用于调试或小规模测试）
返回:
data_iter: 迭代器 - 生成批量数据的迭代器
src_vocab: Vocab - 源语言（输入）的词表对象
tgt_vocab: Vocab - 目标语言（输出）的词表对象
"""
def load_data_nmt(downloader, batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt(downloader)) # 下载并预处理原始数据
    # source: 源语言单词列表
    # target: 目标语言单词列表
    source, target = tokenize_nmt(text, num_examples) # 词元化 分词并截取指定数量的样本
    # 构建 源语言词表（src_vocab）
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 构建 目标语言词表（tgt_vocab）
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 将语言序列转换为张量并计算有效长度(将机器翻译的文本序列转换成小批量)
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 组合数据为元组 (源语言ID, 源语言有效长度, 目标语言ID, 目标语言有效长度)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size) # 创建数据迭代器
    return data_iter, src_vocab, tgt_vocab # 返回迭代器和词表


# 获取《时光机器》的 词元索引序列和词表对象
# max_tokens：限制返回的词元索引序列的最大长度（默认 -1 表示不限制）
def load_corpus_time_machine(downloader, max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine(downloader) # 加载文本数据，得到文本行列表
    tokens = tokenize(lines, 'char') # 词元化：文本行列表→词元列表，按字符级拆分
    vocab = Vocab(tokens) # 构建词表
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    # vocab[token] 查询词元的索引（若词元不存在，则返回0，即未知词索引）
    # corpus：list，每个元素为词元的对应索引
    corpus = [vocab[token] for line in tokens for token in line] # 展平词元并转换为索引
    if max_tokens > 0: # 限制词元序列长度
        corpus = corpus[:max_tokens] # 截断 corpus 到前 max_tokens 个词元
    # corpus：词元索引列表（如 [1, 2, 3, ...]）
    # vocab：Vocab对象，用于管理词元与索引的映射
    return corpus, vocab


# 数据生成器：【随机采样】从长序列中随机抽取子序列，生成小批量数据
# batch_size：指定每个小批量中子序列样本的数目
# num_steps：每个子序列中预定义的时间步数(每个子序列长度)
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    # 随机范围若超过[0,num_steps-1]，则从num_steps开始，往后都会与已有的重复，且少了开头的部分子序列
    # random.randint(0, num_steps-1) 生成一个随机整数offset，范围是[0, num_steps-1]
    # corpus[random.randint(0, num_steps - 1):]截取从该偏移量到序列末尾的子序列
    corpus = corpus[random.randint(0, num_steps - 1):] # 随机偏移起始位置
    # 减去1，是因为需要考虑标签，标签是右移一位的序列
    num_subseqs = (len(corpus) - 1) // num_steps # 总可用 子序列数

    # 生成随机起始索引：长度为num_steps 的子序列 的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps)) # 起始索引列表
    # 在随机抽样的迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices) # 随机打乱顺序

    def data(pos): # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    # 序列长度35，时间步数5，则最多可有(35-1)/5=34/5=6个子序列
    # 批量大小2，则可生成批量数=6个子序列/批量大小2=3个小批量
    num_batches = num_subseqs // batch_size # 可生成的小批量数=总可用子序列数÷批量大小
    # 构造小批量数据(每次取batch_size个随机起始索引，生成输入X和标签Y)
    # i就是 当前批量在 总子序列中的第几批开头位置
    # 从已有的 打乱好的 起始索引list中，选出当前批量对应的那个下标位置上 的起始索引
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size] # 每批次对应的起始索引
        X = [data(j) for j in initial_indices_per_batch]     # 输入子序列
        Y = [data(j + 1) for j in initial_indices_per_batch] # 标签(右移一位)
        yield torch.tensor(X), torch.tensor(Y) # 使用yield实现生成器，节省内存


# 数据生成器：【顺序分区】按顺序划分长序列，生成小批量数据，保证完整覆盖序列
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps) # 随机偏移起始位置
    # 确保能整除 batch_size，避免最后一个小批量不足
    # (len(corpus) - offset - 1) 起始位置偏移后，剩余右侧 所需的最少长度
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size # 有效词元数
    # 重构为批量优先格式：将序列重塑为 (batch_size批量大小, sequence_length序列长度) 的张量，便于批量处理
    # sequence_length序列长度：每个样本（序列）的时间步数（或词元数）
    Xs = torch.tensor(corpus[offset: offset + num_tokens]) # 截取有效词元区域，这里得到的向量形式的张量
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens]) # 可作为标签的有效词元区域
    # 重塑张量形状，每列皆为一个批量，每行皆为单批量的序列长度 即总词元数大小
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps # 批量数=/每个小批量的时间步数 即序列长度
    # 按步长分割小批量：沿序列长度维度（axis=1）滑动窗口，生成连续的小批量
    # 将单次批量的总序列大小分割为多个子序列
    for i in range(0, num_steps * num_batches, num_steps):
        # 从第i列开始，取num_steps列
        X = Xs[:, i: i + num_steps] # 输入子序列
        Y = Ys[:, i: i + num_steps] # 标签
        yield X, Y # 使用yield实现生成器


# 数据加载器类：将随机采样和顺序分区包装到一个类中，以便稍后可以将其用作数据迭代器
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    # max_tokens：限制返回的词元索引序列的最大长度（默认 -1 表示不限制）
    def __init__(self, downloader, batch_size, num_steps, use_random_iter, max_tokens):
        # 初始化选择采样方式
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random     # 随机取样
        else:
            self.data_iter_fn = seq_data_iter_sequential # 顺序分区
        self.corpus, self.vocab = load_corpus_time_machine(downloader, max_tokens) # 加载语料和词表
        self.batch_size, self.num_steps = batch_size, num_steps

    # __iter__实现迭代器协议：使对象可迭代，直接用于for循环
    # 从语料库(self.corpus)中 按指定的batch_size和num_step（即sequence_length） 生成批量数据
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

'''
数据加载函数：同时返回数据迭代器和词表
batch_size     ：每小批量的子序列数量
num_steps      ：每个子序列的时间步数（词元数）
use_random_iter：是否使用随机采样（默认顺序分区）
max_tokens     ：限制语料库的最大词元数
    返回值
data_iter：SeqDataLoader 实例（可迭代）
vocab    ：词表对象（用于词元与索引的映射）
'''
def load_data_time_machine(downloader, batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        downloader, batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# 编码器类（抽象基类）
# 长度可变序列 → 固定形状的编码状态
# 其输出是 输入序列的压缩表示
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        # Encoder 继承自 nn.Module（PyTorch 的基类）
        # super(Encoder, self)：从 Encoder的父类（即 nn.Module）开始查找方法
        # **kwargs 表示接收任意数量的关键字参数
        super(Encoder, self).__init__(**kwargs)
        # 初始化父类nn.Module，支持任意关键字参数

    # 将输入序列编码为固定形状的状态
    def forward(self, X, *args):
        """ 前向传播接口（需子类实现）
        X: 输入序列（如源语言句子）
        *args: 可变参数（如输入序列长度、掩码等）
        返回: 编码后的状态（具体形式由子类决定）
        """
        raise NotImplementedError # 强制子类必须实现此方法

# 解码器类（抽象基类）
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        # 初始化父类nn.Module，支持任意关键字参数

    # 利用编码器输出，来初始化解码状态(对压缩好的输入序列表示 做处理，例如调整格式或筛选)
    # 即 将编码器的输出enc_outputs 转换为解码器所需的 解码前要的状态
    def init_state(self, enc_outputs, *args):
        """ 初始化解码状态接口（需子类实现）
        enc_outputs: 编码器的输出结果
        *args: 可变参数（如编码器隐藏状态等）
        返回: 初始化解码状态（如RNN的初始隐藏状态）
        """
        raise NotImplementedError # 强制子类必须实现此方法

    # 根据当前输入和状态生成输出
    # 即 将输入(如 在前一时间步生成的词元) 和 编码后的状态 映射成当前时间步的输出词元
    def forward(self, X, state):
        """ 解码前向传播接口（需子类实现）
        X: 当前解码输入（如前一个时间步的词）
        state: 解码状态（由init_state或前一时间步生成）
        返回:
            output: 当前时间步的输出（如预测的词分布）
            new_state: 更新后的解码状态
        """
        raise NotImplementedError # 强制子类必须实现此方法

# 编码器-解码器 架构基类
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder  # 编码器实例
        self.decoder = decoder  # 解码器实例

    """ 端到端前向传播
    编码器：压缩输入
    解码器：结合上下文 生成输出
    enc_X: 编码器输入（如 源语言句子）
    dec_X: 解码器输入（如 目标语言句子，训练时使用）
    *args: 可变参数（传递给编码器和解码器）
    流程:
        1. 编码阶段: enc_X → enc_outputs
        2. 初始化解码状态: enc_outputs → dec_state
        3. 解码阶段: (dec_X, dec_state) → 最终输出
    返回: 解码器的输出结果
    """
    def forward(self, enc_X, dec_X, *args):
        # 1. 编码阶段：处理输入序列
        enc_outputs = self.encoder(enc_X, *args)
        # 2. 初始化解码状态（如RNN的初始隐藏状态）
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 3. 解码阶段：生成输出序列
        # 注意：这里假设dec_X是完整的目标序列（训练时使用teacher forcing）
        return self.decoder(dec_X, dec_state)

class AdditiveAttention(nn.Module):
    """ 加性注意力机制（Additive Attention/ Bahdanau Attention）
    通过全连接层和tanh激活计算注意力分数，适用于查询和键维度不同的情况
    这里的类成员变量说的维度皆是指特征维度
    key_size    : 键向量的维度
    query_size  : 查询向量的维度
    num_hiddens : 隐藏层大小（注意力计算中间层的维度）
    dropout     : dropout比率
    kwargs      : 其他传递给父类的参数
    加性注意力通过将查询和键映射到同一空间并相加，然后通过一个单层前馈网络计算注意力分数
    自己理解：
    就是这个工具人小型神经网络中，只有一个隐藏层(全连接层)(用于转相同维度)，和一个输出层(用于转单个标量)
    键和查询皆通过这个神经网络，经隐藏层后映射到相同维度，然后再结合，再经过线性层映射为标量，得到权重值
    结合方式是 广播求和再经过非线性函数tanh。
    得到注意力权重后，再经过随机失活层再与值做批量矩阵乘法得到预测值。
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 将键和查询映射到相同隐藏空间的全连接层
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)     # 键变换矩阵
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)   # 查询变换矩阵
        # 将隐藏状态映射到标量分数的线性变换
        # 输出单分数值的全连接层（相当于注意力权重向量）
        self.w_v = nn.Linear(num_hiddens, 1, bias=False) # 分数计算层
        self.dropout = nn.Dropout(dropout)  # 注意力权重dropout，防止过拟合

    def forward(self, queries, keys, values, valid_lens):
        '''
        queries     : 查询向量 [batch_size, 查询个数num_queries, query_size]
        keys        : 键向量   [batch_size, 键值对个数num_kv_pairs, key_size]
        values      : 值向量   [batch_size, 键值对个数num_kv_pairs, value_size]
        valid_lens  : 有效长度 [batch_size,] 或 [batch_size, num_queries] 用于掩码处理
        返回：注意力加权后的值向量 [batch_size, num_queries, value_size]
        '''
        # 投影变换：将查询和键映射到相同维度的隐藏空间
        queries, keys = self.W_q(queries), self.W_k(keys) # 形状变为 [batch_size, *, num_hiddens]
        # （核心步骤）
        # 以便进行广播相加，在维度扩展后：
        # queries 添加中间维度 -> [batch_size, num_queries, 1, num_hiddens]
        # keys    添加第二维度 -> [batch_size, 1, num_kv_pairs, num_hiddens]
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 广播方式求和，使每个查询与所有键相加，
        # 自动扩展为 [batch_size, 查询数num_queries, 键值对数num_kv_pairs, num_hiddens]
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # 创建每个查询-键对的联合表征
        features = torch.tanh(features) # 非线性变换(原始论文使用tanh)，保持形状不变

        # 通过线性层计算注意力分数（原始论文的a(s,h)计算）
        # 通过w_v将特征映射为标量分数 -> [batch_size, num_queries, num_kv_pairs, 1]
        # 移除最后一个维度 -> [batch_size, num_queries, num_kv_pairs]
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1) # 将特征映射为一个标量

        # 应用掩码softmax获取归一化注意力权重
        # 只保留有效位置的数据然后进行归一化
        self.attention_weights = masked_softmax(scores, valid_lens)

        # 注意力对值进行加权求和（使用dropout正则化）
        # torch.bmm批矩阵乘法：
        # [batch_size, num_queries, num_kv_pairs] × [batch_size, num_kv_pairs, value_size]
        # 结果形状 -> [batch_size, num_queries, value_size]
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    """ 缩放点积注意力（Scaled Dot-Product Attention）
    通过点积计算查询和键的相似度，并用缩放因子稳定训练过程
    dropout: 注意力权重的随机失活率
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout) # 初始化随机失活层

    def forward(self, queries, keys, values, valid_lens=None):
        '''
        queries   : (batch_size，查询的个数，d)
        keys      : (batch_size，“键－值”对的个数，d)
        values    : (batch_size，“键－值”对的个数，值的维度)
        valid_lens: (batch_size，)或(batch_size，查询的个数)
        return:注意力加权后的值向量 [batch_size, num_queries, value_dim]
        '''
        d = queries.shape[-1] # 获取查询和键的特征维度d（用于缩放）

        # 1：计算点积分数（原始注意力分数）
        # 设置transpose_b=True为了交换keys的最后两个维度
        # keys.transpose(1,2)：将键的最后两维转置，即 从(b,nk,d)变为(b,d,nk)
        # 数学等价：对每个查询向量q和键向量k计算q·k^T 即 q乘以k的转置
        # 缩放因子：/ math.sqrt(d)防止高维点积数值过大，避免梯度消失
        # 矩阵乘法：查询q形状(b,nq,d) × k转置形状(b,d,nk) → 点积分数scores形状(b,nq,nk)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)

        # 2：应用掩码softmax归一化
        # 根据有效长度屏蔽无效位置，并归一化得到注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)

        # 3：注意力权重与值向量加权求和
        # 注意力权重(在训练时会随机失活)与值做批量矩阵乘法
        # 矩阵乘法：weights形状(b,nq,nk) × values形状(b,nk,vd) → 输出形状(b,nq,vd)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 用于序列到序列学习（seq2seq）的循环神经网络编码器
class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层:获得输入序列中每个词元的特征向量
        # 将词元索引转换为密集向量（vocab_size → embed_size）
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 循环神经网络（GRU）
        # embed_size: 输入特征维度（嵌入层输出维度）
        # num_hiddens: 隐状态维度
        # num_layers: 堆叠的RNN层数
        # dropout: 层间dropout概率（仅在num_layers>1时生效
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args): # 前向传播逻辑
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X) # 嵌入层处理：(bch_sz, num_steps)→ (bch_sz, num_steps, embed_size)

        # permute(1, 0, 2)交换前两维，即 时间步和批次维度
        # 因为在循环神经网络模型中，轴一对应于时间步
        # 即 RNN要求输入形状为 (num_steps, batch_size, embed_size)
        X = X.permute(1, 0, 2) # 形状变为 (num_steps, batch_size, embed_size)

        # RNN处理
        # 默认初始隐状态state=None时，自动初始化为全零
        # 若未提及状态，则默认为0
        output, state = self.rnn(X)

        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state # 所有时间步的隐状态，最后一层的最终隐状态

class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        """注意力权重属性，
        子类必须实现此方法
        以返回注意力权重（可视化对齐关系）"""
        raise NotImplementedError # 强制子类必须实现此方法

class Seq2SeqAttentionDecoder(AttentionDecoder):
    """ 带有注意力机制的 序列到序列解码器 """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        """ 初始化注意力解码器
        vocab_size : 词汇表大小
        embed_size : 词嵌入维度 (每个数据被表示为的向量维度)
        num_hiddens: 隐藏层维度 (单个隐藏层的神经元数量)
        num_layers : RNN层数   (隐藏层的堆叠次数)
        dropout    : 随机失活率
        """
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)

        # 加性注意力模块：查询/键/值 的维度均为num_hiddens
        # 实现相似度计算与权重分配
        self.attention = AdditiveAttention(
            num_hiddens,  # 查询向量维度
            num_hiddens,  # 键向量维度
            num_hiddens,  # 值向量维度
            dropout) # 实现Bahdanau的加性注意力机制 (通过联合表征得到注意力权重)

        # 词嵌入层：将词元ID映射为向量 (将词元映射为指定维度的向量)
        self.embedding = nn.Embedding(vocab_size, embed_size) #（词元ID → 向量）

        # GRU(门控循环单元)循环神经网络层（输入=词嵌入+上下文向量，输出隐藏状态）
        # 体现注意力对解码的引导
        self.rnn = nn.GRU(
            embed_size + num_hiddens, # 输入维度为 词嵌入+上下文向量  即 当前输入和上下文变量
            num_hiddens,  # 隐藏层维度
            num_layers,   # 堆叠层数
            dropout=dropout) #（输入=词嵌入+上下文向量）

        # 输出全连接层：将隐藏状态(RNN输出)映射回 词汇表空间(词汇表维度)
        self.dense = nn.Linear(num_hiddens, vocab_size) # （隐藏状态→词表概率分布）

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        ''' 处理编码器输出，准备解码器初始状态
        enc_outputs: 编码器的输出（所有时间步的隐状态outputs, 最后一层的最终隐状态hidden_state）
                    即(batch_size, num_steps, num_hiddens)
        enc_valid_lens:编码器有效长度 (batch_size,)
        返回：解码器初始状态，三元组 (编码器输出、隐藏状态、有效长度)
            outputs：编码器所有时间步的隐藏状态（用于注意力计算）(已转置，将时间步防御第0维上)
            hidden_state：编码器最终隐藏状态（解码器初始状态）
            enc_valid_lens：源序列的有效长度（掩码处理用）
        '''
        # outputs     形状(batch_size，num_steps ，num_hiddens)（PyTorch的GRU默认输出格式）
        # hidden_state形状(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        # .permute(1,0,2)调整outputs维度为(num_steps, batch_size, num_hiddens)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        '''
        X: 输入序列，形状为(batch_size, num_steps)
        state: 解码器状态，即 初始状态三元组 (enc_outputs, hidden_state, enc_valid_lens)
        返回:
            outputs: 输出序列，形状为(batch_size, num_steps, vocab_size)
            state: 更新后的状态
        '''
        # 解包状态：编码器输出、编码器最终隐藏状态、有效长度
        # enc_outputs 形状(batch_size,num_steps ,num_hiddens)
        # hidden_state形状(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state

        # 词嵌入(将输入词元ID转换为向量)
        # 并调整维度顺序：(batch_size, num_steps, embed_size)→(num_steps, batch_size, embed_size)
        # .permute(1, 0, 2)将第0维和第1维的内容交换位置
        X = self.embedding(X).permute(1, 0, 2) # 输出X的形状(num_steps,batch_size,embed_size)

        outputs, self._attention_weights = [], [] # 存储输出和注意力权重
        for x in X: # 逐时间步处理(解码)
            # 当前隐状态作为查询：首次为编码器中最后一层最终隐状态，后续会动态更新，始终指向最后一层的当前隐状态
            # hidden_state[-1]取解码器最后一个隐藏层状态（最顶层GRU的输出）
            # 取出后  .unsqueeze()在这个最后的隐状态的 第1维位置插入一个大小为1的维度
            query = torch.unsqueeze(hidden_state[-1], dim=1) # 形状(batch_size,1,num_hiddens)

            # 计算上下文向量（Bahdanau注意力的核心）
            # 通过加性注意力计算，权重由 query和编码器状态 enc_outputs决定
            context = self.attention( # 形状(batch_size,1,num_hiddens)
                query,          # 查询向量
                enc_outputs,    # 编码器所有时间步的输出
                enc_outputs,    # 作为键和值
                enc_valid_lens  # 有效长度
            ) # (通过联合表征得到注意力权重)

            # 在特征维度上连结 (拼接 上下文向量和当前词嵌入)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)

            # 调整维度并输入RNN(输入rnn目的是通过递归计算来动态更新隐状态)
            # 通过GRU处理：将上下文向量与当前词嵌入拼接后输入GRU
            # 输入形状 (1, batch_size, embed_size+num_hiddens)
            # .permute(1, 0, 2) 将x变形为(1, batch_size, embed_size+num_hiddens)
            # PyTorch的RNN/GRU层要求输入张量形状必须为(seq_len, batch_size, 每个时间步输入的特征维度input_size)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state) # hidden_state动态更新
            outputs.append(out) # 存储输出

            # 存储当前时间步的注意力权重
            self._attention_weights.append(self.attention.attention_weights)

        # 合并所有时间步的输出并投影到词表空间
        # .cat()从列表中的多个(1,batch_size,num_hiddens)，拼接为(seq_len,batch_size,num_hiddens)的三维张量
        # 即，将解码器各时间步的输出拼接为序列，并最终映射为词元概率分布序列，从而生成完整的输出句子
        # 全连接层变换后，outputs的形状为(num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        ''' 注意力权重访问
        返回：每个解码时间步的注意力权重（用于可视化对齐关系）
        '''
        return self._attention_weights

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads # 并行注意力头的数量
        self.attention = DotProductAttention(dropout) # 放缩点集注意力模块

        # 定义线性投影层（将输入映射到不同子空间）
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)  # Q投影
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)    # K投影
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)  # V投影
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias) # 输出拼接后的线性变换

    def forward(self, queries, keys, values, valid_lens):
        """
        queries   : 查询向量 (batch_size, 查询数, query_size)
        keys      : 键向量 (batch_size, 键值对数, key_size)
        values    : 值向量 (batch_size, 键值对数, value_size)
        valid_lens: 有效长度 (batch_size,) 或 (batch_size, 查询的个数)
        即 q，k，v形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
        """
        # 1. 线性投影：将Q/K/V从原始维度投影到 num_hiddens维度
        # 投影后形状: (batch_size, 查询数/键值对数, num_hiddens)
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # 2. 形状变换（为多头并行计算准备）：用transpose_qkv将张量重塑为多头并行格式
        # 经过变换后，输出的 q，k，v 的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(queries, self.num_heads)
        keys    = transpose_qkv(keys   , self.num_heads)
        values  = transpose_qkv(values , self.num_heads)

        # 3. 处理有效长度（扩展到每个头）：若有valid_lens，将其复制到每个注意力头
        if valid_lens is not None:
            # 作用：将有效长度复制num_heads次并保持维度 (batch_size*num_heads, ...)
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # 4. 并行计算多头注意力：每个头独立计算缩放点积注意力
        # 输出形状:(batch_size*num_heads，查询个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # 5. 逆转形状变换（恢复原始维度）：用transpose_output恢复原始形状
        # 输出拼接后形状: (batch_size, 查询数, num_hiddens)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)

        # 6. 最终线性变换：线性层W_o整合多头信息
        return self.W_o(output_concat)

# 辅助函数：为多头并行计算变换形状
def transpose_qkv(X, num_heads):
    """为了 多注意力头的并行计算 而变换形状
    即 将张量重塑为多头并行计算格式
    示例：
    输入 X：(2, 4, 100)（batch_size=2, seq_len=4, num_hiddens=100）
    输出：(2 * 5, 4, 20)（num_heads=5, depth_per_head=20）
    """
    # 增加num_heads维度
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # 增加num_heads维度

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3) # 每批次数据皆按头数分批，几个头就几批

    # 最终合并batch和num_heads维度: (batch_size*num_heads, 序列长度, 每个头的维度)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


# 辅助函数：逆转transpose_qkv的形状变换
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作
    即 将多头输出恢复为原始维度格式 """
    # 输入形状: (batch_size*num_heads, 序列长度, 每个头的维度)
    # 恢复维度: (batch_size, num_heads, 序列长度, 每个头的维度)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3) # (batch_size, seq_len, num_heads, 每个头的维度depth_per_head)
    # 合并最后两个维度: (batch_size, 序列长度, num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionalEncoding(nn.Module):
    """位置编码模块：为序列注入绝对位置信息
    构造函数中 实现了位置编码的预计算
    在训练/推理时只需根据实际序列长度动态截取，既节省内存又提高效率
    max_len     ：词元在序列中的位置 总数
    num_hiddens ：位置编码的不同维度 总数即总维度
    实例化类对象传入时，num_hiddens作为嵌入维度
    """
    # num_hiddens 嵌入维度
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P（位置嵌入矩阵）
        # 预计算位置编码矩阵（1, max_len, num_hiddens）
        self.P = torch.zeros((1, max_len, num_hiddens)) # 初始化三维张量

        # 生成位置编码的频率分量（基于论文公式）
        # 公式：
        # PE(pos,2j)   = sin(pos/(10000^{2j/d}))
        # PE(pos,2j+1) = cos(...)
        # positions为列向量，表示行i的 位置索引 [0,1,...,max_len-1]
        # dims：分母中10000的幂，
        # arange(0,n,2)从0到n-1，步长为2，生成【偶数】索引序列[0,2,4,...,n-1]，对应公式中的2j
        # /num_hiddens 归一化到[0,1)区间（例如num_hiddens=512时，dims=[0/512, 2/512, …, 510/512]）
        positions = torch.arange(max_len).float().reshape(-1, 1) # 形状(max_len,1)
        dims = torch.arange(0, num_hiddens, 2).float() / num_hiddens  # 维度索引归一化,形状(num_hiddens/2,)
        freqs = torch.pow(10000, dims)  # 10000^(2j/d) 计算频率分母项，形状同dims

        # sin/cos的输入值（pos / (10000^{2j/d})）
        # 广播除法：将positions的每一行除以freqs的所有元素
        # 出来X形状(词元在序列中的位置总数max_len，位置编码的总维度num_hiddens/2)，/2是因为前面步进为2
        # 例如 pos=2, j=0时：X[2][0] = 2 / 10000⁰ = 2
        X = positions / freqs # 即（行i/(10000^{2j/d})），j是列

        # 填充位置编码矩阵：偶数维度用sin，奇数维度用cos
        # 0::2 从0开始步进为2的切片，即 偶数列 0，2，4，8，…
        # 1::2 从1开始步进为2的切片，即 奇数列 1，3，5，7，…
        self.P[:, :, 0::2] = torch.sin(X)  # 所有偶数列填充sin值
        self.P[:, :, 1::2] = torch.cos(X)  # 所有奇数列填充cos值

    # 构造函数里 X=positions/freqs 的 X 在用于生成self.P后即被销毁
    # 前向传播里 X.shape[1] 用的 X 是来自于函数头的参数X，即 实际输入的数据
    def forward(self, X):
        """前向传播：将位置编码加到输入上
        前向传播中的截取操作 pos_embed=self.P[:, :seq_len,:]：动态适配输入序列长度
        核心目的：避免固定max_len导致的内存浪费，实现按需截取。
        """
        # 根据实际序列长度截取位置编码（避免固定max_len导致内存浪费）
        seq_len = X.shape[1] # X.shape[1] 获取输入序列的实际长度
        pos_embed = self.P[:, :seq_len, :].to(X.device)  # 按实际序列长度截取位置编码，移动设备兼容性处理
        X = X + pos_embed # 绝对位置编码：直接相加，将位置编码加到输入上（广播机制自动对齐batch维度）
        return self.dropout(X) # 应用随机失活

# 定义位置前馈网络类（Position-wise Feed-Forward Network）
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络
    特点：对输入序列中每个位置的向量独立进行非线性变换，不涉及位置间交互
    结构：两层全连接层 + ReLU激活函数
    """
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        """
        参数说明：
        ffn_num_input  : 输入特征维度（如词向量维度）
        ffn_num_hiddens: 中间层隐藏单元数（通常远大于输入维度）
        ffn_num_outputs: 输出特征维度
        """
        super(PositionWiseFFN, self).__init__(**kwargs)
        # 第一层全连接：维度扩展（输入维度 → 隐藏维度）
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        # 非线性激活函数（引入非线性能力）
        self.relu = nn.ReLU()
        # 第二层全连接：维度压缩（隐藏维度 → 输出维度）
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """ 前向传播逻辑：
        1. 对输入张量X的每个位置独立应用相同变换
        2. 维度变化： (batch_size, seq_length, ffn_num_input)
                  → (batch_size, seq_length, ffn_num_hiddens)
                  → (batch_size, seq_length, ffn_num_outputs)
        """
        # 执行：线性变换 → 激活函数 → 线性变换
        return self.dense2(self.relu(self.dense1(X)))

# 残差连接 + 层归一化模块
class AddNorm(nn.Module):
    """残差连接后进行层规范化
    实现残差连接（Residual Connection）后进行层归一化（Layer Normalization）
    结构：Sublayer Output → Dropout → (X + Dropout(Y)) → LayerNorm
    作用：
        1. 残差连接：缓解梯度消失，支持深层网络训练
        2. 层归一化：稳定各层输入分布，加速收敛
        3. Dropout在残差连接前应用：防止子层过拟合同时保护残差梯度通路
    """
    def __init__(self, normalized_shape, dropout, **kwargs):
        """
        normalized_shape: 输入张量的最后维度（如[3,4]表示最后维度为4）
        dropout: 随机失活概率
        """
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout) # 作用在子层输出
        # 层归一化层，作用于残差连接结果(对每个样本的所有特征进行归一化)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """ 前向传播
        X: 残差输入（通常来自上一层或跳跃连接）
        Y: 子层输出（如注意力层/前馈网络输出）
        返回：层归一化后的张量
        逻辑：
            1. 子层输出Y经过Dropout（防止过拟合）
            2. 执行残差连接：X + Dropout(Y)
            3. 层归一化：稳定数值分布，加速收敛
        Dropout(Y) 即为 sublayer
        Dropout应作用于：子层输出(Y)之后、残差连接之前
        顺序： Dropout → 残差连接 → 层归一化
        """
        # 残差连接：输入X与子层输出sublayer相加
        # 注意：X和sublayer必须形状相同（如[batch_size, seq_length, dim]）
        return self.ln(self.dropout(Y) + X)






# import matplotlib.pyplot as plt
# import numpy as np
#
# axis_x = np.array([-8, -7, -6, -5, -4, -3, -2, -1])
# axis_y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# fig1 = plt.figure(1)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.plot(axis_x, axis_y)
# plt.draw()
# plt.pause(4)  # 间隔的秒数： 4s
# plt.close(fig1)
#
# fig2 = plt.figure(2)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.plot(axis_y, axis_x)
# plt.draw()
# plt.pause(6)  # 间隔的秒数：6s
# plt.close(fig1)

