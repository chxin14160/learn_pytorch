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


def learn_padding_and_stride():
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

# learn_padding_and_stride()


def learn_multi_in_out():
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


    '''
    1x1 卷积的本质：
    1x1 卷积相当于对每个空间位置 (i,j) 独立进行 全连接层计算（跨通道的线性变换）
    可以通过矩阵乘法高效实现
    '''
    # 1x1 卷积的多输入多输出通道版本
    def corr2d_multi_in_out_1x1(X, K):
        c_i, h, w = X.shape          # 输入数据的形状: (输入通道数, 高度, 宽度)
        c_o = K.shape[0]             # 输出通道数 (K的第一维)
        print(f"展平前 X 的形状：{X.shape}")
        print(f"展平前 核张量X：\n{X}")
        print(f"展平前 核张量K 的形状：{K.shape}")
        print(f"展平前 核张量K：\n{K}")
        X = X.reshape((c_i, h * w))  # 将空间维度展平: (c_i, h*w)
        K = K.reshape((c_o, c_i))    # 将卷积核展平: (c_o, c_i)
        print(f"展平后 X 的形状：{X.shape}")
        print(f"展平后 核张量X：\n{X}")
        print(f"展平后 核张量K 的形状：{K.shape}")
        print(f"展平后 核张量K：\n{K}")
        # 全连接层中的矩阵乘法
        Y = torch.matmul(K, X)       # 矩阵乘法: (c_o, h*w)
        return Y.reshape((c_o, h, w)) # 恢复空间维度: (c_o, h, w)

    # 验证与通用卷积的一致性
    # 从均值=0，方差=1的离散正态分布中随机抽取数值
    X = torch.normal(0, 1, (3, 3, 3))    # 组成3*3*3的张量
    K = torch.normal(0, 1, (2, 3, 1, 1)) # 组成2*3*1*1的张量

    print(f"---------------------")
    Y1 = corr2d_multi_in_out_1x1(X, K)  # 1*1卷积优化计算
    Y2 = corr2d_multi_in_out(X, K)      # 通用多通道卷积
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6 # 验证结果一致

# learn_multi_in_out()


def learn_pooling():
    # 自定义二维汇聚层
    def pool2d(X, pool_size, mode='max'):
        p_h, p_w = pool_size # 池化窗口的高度和宽度
        # 计算输出张量的形状（输入大小 - 池化窗口大小 + 1）
        Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                window = X[i: i + p_h, j: j + p_w] # 提取当前窗口的子矩阵
                if mode == 'max':   # 最大汇聚：取窗口内的最大值
                    Y[i, j] = window.max()
                elif mode == 'avg': # 平均汇聚：取窗口内的平均值
                    Y[i, j] = window.mean()
        return Y

    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(f"输入X：\n{X}")
    print(f"最大汇聚层（最大池化层）：\n{pool2d(X, (2, 2))}")
    print(f"平均汇聚层（平均池化层）：\n{pool2d(X, (2, 2), 'avg')}")


    # 直接使用 PyTorch中内置的 二维最大汇聚层：
    # 输入输出通道数皆为1的 4*4张量
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(f"输入X：\n{X}")

    pool2d = nn.MaxPool2d(3)  # 池化窗口大小 3×3
    print(f"最大汇聚层（最大池化层）：池化窗口大小 3×3"
          f"\n{pool2d(X)}")

    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(f"最大汇聚层（最大池化层）：输入四周补1圈0，步长设为2\n{pool2d(X)}")

    pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
    print(f"最大汇聚层（最大池化层）：非对称窗口和步长"
          f"\n步长为2行3列，仅在宽度方向补1圈0："
          f"\n{pool2d(X)}")


    # 多个通道的情况：
    # 张量的维度顺序通常遵循 NCHW 格式（批量大小 × 通道数 × 高度 × 宽度）
    X = torch.cat((X, X + 1), 1) # 沿现有维度 第一维即通道 拼接两个张量
    print(f"输入X：通道数为2\n{X}")

    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(f"最大汇聚层（最大池化层）：输入四周补1圈0，步长设为2"
          f"\n池化后通道数不变"
          f"\n{pool2d(X)}")

# learn_pooling()


import common

net = nn.Sequential( # 定义顺序容器
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), # 输入通道=1，输出通道=6，5×5 卷积核，padding=2 保持尺寸不变
    nn.AvgPool2d(kernel_size=2, stride=2),  # 2×2平均池化，步幅=2（输出尺寸减半）
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 输入通道=6，输出通道=16，5×5卷积核（无padding，尺寸缩小）
    nn.AvgPool2d(kernel_size=2, stride=2),  # 2×2平均池化，步幅=2（输出尺寸减半）
    nn.Flatten(),                           # 展平层：将多维张量展平为一维向量，供全连接层使用
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(), # 经过两次池化后，特征图尺寸为(16,5,5)，展平为 16*5*5=400 维向量
    # 后续接两个隐藏层（120 和 84 个神经元）和输出层（10 类）
    nn.Linear(120, 84), nn.Sigmoid(), # Sigmoid激活函数，将输出压缩到(0,1)区间（现代CNN通常用ReLU）
    nn.Linear(84, 10))  # 输出层通常不用激活函数，CrossEntropyLoss会包含Softmax

# 模拟输入：batch_size=1，通道=1，高度=28，宽度=28（Fashion-MNIST尺寸）
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net: # 逐层前向传播并打印输出形状
    X = layer(X)  # 前向计算
    print(layer.__class__.__name__,'output shape: \t',X.shape) # 打印层类型和输出形状（张量的维度）

batch_size = 256 # 批量大小，每次处理256张图像
# 加载Fashion-MNIST数据集，返回 训练集和测试集的DataLoader对象
train_iter, test_iter = common.load_data_fashion_mnist(batch_size=batch_size)

# 评估函数
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):  # 判断模型是否为深度学习模型
        net.eval()  # 设置为评估模式（关闭Dropout和BatchNorm的随机性）
        if not device: # 如果没有指定设备，自动使用模型参数所在的设备（如GPU）
            device = next(iter(net.parameters())).device # 自动检测设备
    # 初始化计数器：累计 正确预测的数量 和 总预测的数量
    metric = common.Accumulator(2) # metric[0]=正确数, metric[1]=总数
    with torch.no_grad():  # 禁用梯度计算（加速评估并减少内存占用）
        for X, y in data_iter:  # 每次从迭代器中拿出一个X和y
            # 将数据移动到指定设备（如GPU）
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # 计算预测值和准确率，并累加到metric中
            metric.add(common.accuracy(net(X), y), y.numel()) # 累加准确率和样本数
    # metric[0, 1]分别为网络预测正确数量和总预测数量
    return metric[0] / metric[1] # 计算准确率

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) # Xavier初始化，保持输入输出的方差稳定
    net.apply(init_weights)  # 应用初始化到整个网络（初始化权重）
    print('training on', device)
    net.to(device)  # 模型移至指定设备（如GPU）
    optimizer = torch.optim.SGD(net.parameters(), lr=lr) # 定义优化器：随机梯度下降（SGD），学习率为lr
    loss = nn.CrossEntropyLoss()  # 交叉熵损失
    # 初始化动画绘图器，用于动态绘制训练曲线
    animator = common.Animator(xlabel='epoch',
                               xlim=[1, num_epochs],
                               legend=['train loss', 'train acc', 'test acc'])
    # 初始化计时器和计算总批次数
    timer, num_batches = common.Timer(), len(train_iter)
    # 开始训练循环
    for epoch in range(num_epochs):
        # Accumulator(3)创建3个变量：训练损失总和、训练准确度总和、样本数
        metric = common.Accumulator(3) # 用于跟踪训练损失、准确率和样本数
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
                metric.add(l * X.shape[0], common.accuracy(y_hat, y), X.shape[0])
            timer.stop()            # 停止计时
            train_l = metric[0] / metric[2]     # 平均训练损失
            train_acc = metric[1] / metric[2]   # 平均训练准确率
            # 每训练完1/5的epoch 或 最后一个batch时，更新训练曲线
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 测试集准确率
        animator.add(epoch + 1, (None, None, test_acc)) # 更新测试集准确率曲线
    print(f'最终结果：loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'训练速度（样本数/总时间）：{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

lr, num_epochs = 0.9, 10 # 学习率，训练轮数(训练10轮)
train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu())
