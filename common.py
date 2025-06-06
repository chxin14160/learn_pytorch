import time
import numpy as np
from IPython import display
from torch.utils.data import DataLoader, TensorDataset
import torch

# import matplotlib
# # 强制使用 TkAgg 或 Qt5Agg 后端 (使用独立后端渲染)
# matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'，根据你的系统安装情况
# # matplotlib.use('Qt5Agg')  # 或者使用 'Qt5Agg'，根据你的系统安装情况
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor


# import matplotlib
# matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'，根据你的系统安装情况


def load_data_fashion_mnist(batch_size, resize=None):
    # Download data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",  # 数据集存储的位置
        train=True,  # 加载训练集（True则加载训练集）
        download=True,  # 如果数据集在指定目录中不存在，则下载（True才会下载）
        transform=ToTensor(),  # (使用什么格式转换,这里是对图片进行预处理，转换为tensor格式) 应用于图像的转换列表，例如转换为张量和归一化
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,  # 加载测试集（False则加载测试集）
        download=True,
        transform=ToTensor(),
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

