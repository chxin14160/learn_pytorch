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



# 构造一个二维卷积层，输出通道为1，且卷积核形状为（1，2）
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8)) # 输入形状：[batch_size=1, channels=1, height=6, width=8]
Y = Y.reshape((1, 1, 6, 7)) # 目标形状：[batch_size=1, channels=1, height=6, width=7]
lr = 3e-2  # 学习率=3*10的-2次方=0.03
print(f"输入：\n{X}")
print(f"目标：\n{Y}")

for i in range(10):
    Y_hat = conv2d(X)       # 结果卷积层的预测结果值(前向传播)
    l = (Y_hat - Y) ** 2    # 计算损失值(均方差MSE损失)
    conv2d.zero_grad()      # 将梯度清零
    l.sum().backward()      # 反向传播
    # 梯度下降，更新权重 (更新可学习的参数)
    # 迭代卷积核
    # 可以使用优化器（如 torch.optim.SGD）替代手动更新
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0: # 每2轮打印一次损失
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(f"最终卷积核权重：\n{conv2d.weight.data.reshape((1, 2))}")


def padding_and_stride():
    # 为了方便起见，这里定义了一个计算卷积层的函数
    # 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
    def comp_conv2d(conv2d, X):
        # 这里的（1，1）表示批量大小和通道数都是1
        # 转换为 4D 张量，格式为 (batch_size, channels, height, width)
        X = X.reshape((1, 1) + X.shape) # (1, 1, 8, 8)
        Y = conv2d(X) # 执行卷积运算，输出 Y 的形状为 (1, 1, H_out, W_out)
        # 省略前两个维度：批量大小和通道，返回 (H_out, W_out)
        return Y.reshape(Y.shape[2:])

    # 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
    # padding=1（每边填充 1 行/列，总填充量为 2 行/列）
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) # 卷积层定义
    X = torch.rand(size=(8, 8))
    print(f"形状：\n{comp_conv2d(conv2d, X).shape}")

    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    print(f"使用5*3的卷积核，顶部和底部填充2行，宽度两边填充1行，\n"
          f"卷积后输出形状：\n{comp_conv2d(conv2d, X).shape}")

    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    print(f"使用3*3的卷积核，每边填充 1 行/列，宽高步幅皆设为2，从而将输入的宽高皆减半\n"
          f"卷积后输出形状：\n{comp_conv2d(conv2d, X).shape}")


    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(f"使用3*5的卷积核，顶部和底部不填充，左右各填充1行，高度步幅设为3，宽度步幅设为4\n"
          f"卷积后输出形状：\n{comp_conv2d(conv2d, X).shape}")

# padding_and_stride()


# 多输入通道的二维互相关运算
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    # corr2d(x, k) 调用前面定义的二维互相关运算
    # zip(X, K) 将输入数据和卷积核按通道配对，生成一个可迭代的元组序列
    # corr2d(x,k) for x,k in zip(X,K) 生成器表达式：惰性计算每个通道的互相关结果，避免中间存储
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(f"多输入通道互相关运算结果：\n{corr2d_multi_in(X, K)}")
print(f"原始 核张量K 的形状：{K.shape}")
print(f"原始 核张量K：\n{K}")


# 计算多个通道的输出 的互相关
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# 通过将核张量K与K+1（K中每个元素加）和K+2连接起来，构造了一个具有个输出通道的卷积核
# 将三个张量沿 新的第 0 维 堆叠，形成一个3通道卷积核
K = torch.stack((K, K + 1, K + 2), 0)
print(f"核张量K 的形状：{K.shape}")
print(f"核张量K：\n{K}")

print(f"多个通道的输出 的互相关 运算结果：\n{corr2d_multi_in_out(X, K)}")


# 1x1 卷积的多输入多输出通道版本
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape          # 输入形状: (输入通道数, 高度, 宽度)
    c_o = K.shape[0]             # 输出通道数 (K的第一维)
    X = X.reshape((c_i, h * w))  # 将空间维度展平: (c_i, h*w)
    K = K.reshape((c_o, c_i))    # 将卷积核展平: (c_o, c_i)
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)       # 矩阵乘法: (c_o, h*w)
    return Y.reshape((c_o, h, w)) # 恢复空间维度: (c_o, h, w)


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
