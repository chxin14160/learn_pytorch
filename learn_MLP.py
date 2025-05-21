import torch
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
# 通过ReLU函数激活（将所有负值置为0，正值保持不变）
y = torch.relu(x)
# detach() 用于从计算图中分离张量，以便转换为numpy数组进行绘图
plot(x.detach(), y.detach(), 'x', 'ReLU(x)', figsize=(5, 2.5))
# 绘图展示了ReLU函数的典型形状：在x<0时输出0，x≥0时输出x

# 绘制 ReLU函数的导数
# 执行反向传播
# torch.ones_like(x)创建一个与x形状相同的全1张量，作为反向传播的梯度输入。假设y的输出对后续计算的贡献都是1
# retain_graph=True 在反向传播后保留计算图，以便后续还可再进行反向传播。在多次反向传播或计算高阶导数时很有用
y.backward(torch.ones_like(x), retain_graph=True)

# 绘图  x.grad是x的梯度，通过反向传播计算得到
plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))



