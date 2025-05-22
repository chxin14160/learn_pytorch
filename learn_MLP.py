import torch
from torch import nn
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
batch_size = 256

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



''' 多层感知机的从零开始实现 结束 '''


