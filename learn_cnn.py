import torch
from torch import nn


# 二维互相关运算
def corr2d(X, K):  # @save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]): # 遍历输出高度方向
        for j in range(Y.shape[1]): # 遍历输出宽度方向
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
            # i:i + h: 这是第一个维度的切片范围
            # j:j + w: 这是第二个维度的切片范围
            # X[i:i + h, j:j + w]，从输入中提取与卷积核大小相同的区域
            # 将该区域与卷积核K 逐元素相乘
            # 对乘积结果求和，得到输出位置(i, j)的值
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
out = corr2d(X, K)
print(f"输入X：\n{X}")
print(f"卷积核K：\n{K}")
print(f"互相关运算输出：\n{out}")



class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__() # 调用父类 nn.Module 的初始化方法
        # nn.Parameter，将张量注册为可训练参数，这些参数会在训练过程中通过梯度下降进行更新
        self.weight = nn.Parameter(torch.rand(kernel_size)) # 卷积核的权重 初始化为随机张量
        self.bias = nn.Parameter(torch.zeros(1)) # 偏置初始化为零张量

    def forward(self, x):
        # 调用前面定义的corr2d函数，计算输入 x 和 卷积核self.weight 之间的二维互相关运算
        return corr2d(x, self.weight) + self.bias



X = torch.ones((6, 8))  # 构造一个6×8像素的黑白图像
X[:, 2:6] = 0           # 中间四列为黑色（0），其余像素为白色（1）
print(f"{X}")

K = torch.tensor([[1.0, -1.0]])
print(f"高度为1、宽度为2的卷积核K：\n{K}")

Y = corr2d(X, K)
print(f"输入和卷积核 做 互相关运算输出：\n{Y}")

out = corr2d(X.t(), K)
print(f"将输入的二维图像转置，再进行如上的互相关运算：\n{out}")



# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')












