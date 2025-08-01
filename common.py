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
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(5, 2.5), axes=None):
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

def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


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

    # __len__用于定义对象的长度行为。对类的实例调用len()时，Python会自动调用该实例的__len__方法
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
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # 初始化选择采样方式
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random     # 随机取样
        else:
            self.data_iter_fn = seq_data_iter_sequential # 顺序分区
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens) # 加载语料和词表
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
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab









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

