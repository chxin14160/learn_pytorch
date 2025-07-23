import torch
from torch import nn
from torch.nn import functional as F
import common


def learn_AlexNet():
    net = nn.Sequential( # 定义顺序容器
        # 这里使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),

        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))

    # 模拟输入：batch_size=1，通道=1，高度=224，宽度=224（为了适配AlexNet架构）
    X = torch.randn(1, 1, 224, 224)
    for layer in net:   # 逐层前向传播并打印输出形状
        X=layer(X)      # 前向计算
        print(layer.__class__.__name__,'output shape:\t\t',X.shape)

    batch_size = 128 # 每次处理128张图片
    # 加载Fashion-MNIST数据集，返回 训练集和测试集的DataLoader对象
    train_iter, test_iter = common.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10 # 学习率0.01，训练10轮
    common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu())

# learn_AlexNet()


def learn_VGG():
    """
        定义了一个名为vgg_block的函数来实现一个VGG块：
        1、卷积层的数量   num_convs
        2、输入通道的数量 in_channels
        3、输出通道的数量 out_channels
    """
    def vgg_block(num_convs, in_channels, out_channels):
        layers = [] # 创建空网络结果，之后通过循环操作使用append函数进行添加
        for _ in range(num_convs): # 循环操作，添加卷积层和非线性激活层ReLU
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels # 更新输入通道数为当前输出通道数
        # 最后添加最大值汇聚层
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)  # 将列表转换为顺序模型

    # 示例：构建一个包含2个卷积层的VGG块
    block = vgg_block(num_convs=2, in_channels=64, out_channels=128)
    print(f"VGG块的网络结构：\n{block}")

    '''
    原VGG网络有5个卷积块：
        前两个有一个卷积层，输出通道数逐步增加（64→128）
        后三个块有两个卷积层，输出通道数继续增加（256→512→512）
    全连接层固定为3层（4096→4096→10），中间使用 Dropout(0.5) 防止过拟合
    该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11
    '''
    # (卷积层数，输出通道数)
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) # VGG-11架构配置

    # 卷积块通过 conv_arch 动态生成，便于调整架构
    # 假设输入图像经过卷积和池化后，特征图尺寸为 7x7（需根据输入尺寸验证）
    def vgg(conv_arch):
        # 定义空网络结构
        conv_blks = []  # 存储所有卷积块
        in_channels = 1 # 初始输入通道数（如MNIST为1，ImageNet为3）

        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            # 添加vgg块
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            # 下一层输入通道数=当前层输出通道数
            in_channels = out_channels

        # 前面循环结束后，out_channels的值会保留为conv_arch中最后一个元组的第二个元素，即512
        return nn.Sequential( # 组合完整网络
            *conv_blks, nn.Flatten(), # 展平特征图
            # 全连接层部分（经典VGG-11配置）
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10))  # 输出10类（如Fashion-MNIST）

    net = vgg(conv_arch)

    # 构建一个高度和宽度为224的单通道数据样本（模拟224x224单通道输入，如MNIST），以观察每个层输出的形状
    X = torch.randn(size=(1, 1, 224, 224))
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__,'output shape:\t',X.shape) # 每一层的输出形状

    # 构建了一个通道数较少的网络，足够用于训练Fashion-MNIST数据集
    ratio = 4 # 通道数缩减比例
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch] # 通道数整除4，//为整除
    net = vgg(small_conv_arch) # 构建轻量级网络

    lr, num_epochs, batch_size = 0.05, 10, 128 # 学习率=0.05，训练10轮，每轮处理128张图
    train_iter, test_iter = common.load_data_fashion_mnist(batch_size, resize=224)
    common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu())
# learn_VGG()


def learn_NiN():
    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), # 普通卷积层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),     # 1*1卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())     # 1*1卷积

    # 示例：构建一个输入通道=64，输出通道=128的NiN块
    block = nin_block(in_channels=64, out_channels=128, kernel_size=5, strides=1, padding=2)
    print(f"NiN块的网络结构：\n{block}")

    net = nn.Sequential(
        # 第一阶段
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),

        # 第二阶段
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),

        # 第三阶段
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),

        # 分类器部分
        nn.Dropout(0.5), # 失活率=0.5的随机失活，防止过拟合
        # 标签类别数是10，输出通道10(分10类)
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)), # 全局平均池化，用于替代全连接层，减少参数量
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        nn.Flatten())

    # 网络前向传播测试
    X = torch.rand(size=(1, 1, 224, 224)) # 创建一个随机输入张量（模拟224x224的灰度图像）
    for layer in net:
        X = layer(X) # 逐层通过网络
        print(layer.__class__.__name__,'output shape:\t\t', X.shape) # 每层的输出形状

    lr, num_epochs, batch_size = 0.1, 10, 128 # 学习率，训练轮数，(批量大小)每轮训练图像张数
    train_iter, test_iter = common.load_data_fashion_mnist(batch_size, resize=224)
    common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu())

# learn_NiN()


def learn_GoogLeNet():
    class Inception(nn.Module):
        # c1--c4是每条路径的输出通道数
        def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
            super(Inception, self).__init__(**kwargs)
            # 线路1，单1x1卷积层
            self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
            # 线路2，1x1卷积层 后接 3x3卷积层
            self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
            self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
            # 线路3，1x1卷积层 后接 5x5卷积层
            self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
            self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
            # 线路4，3x3最大汇聚层 后接 1x1卷积层
            self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

        def forward(self, x): # 通过不同大小的卷积核捕捉多尺度特征，最后在通道维度拼接
            p1 = F.relu(self.p1_1(x))
            p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
            p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
            p4 = F.relu(self.p4_2(self.p4_1(x)))
            # 在通道维度上连结输出
            return torch.cat((p1, p2, p3, p4), dim=1)

    # 网络整体构架：由5个模块(b1-b5)串联组成：
    # 模块b1：初始卷积
    # 通过大步长(2)和池化层快速降低空间维度
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 输出: 64@48x48
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))      # 输出: 64@24x24

    # 模块b2：中间卷积
    # 使用1x1卷积降维后 再用3x3卷积提取特征
    # 进一步通过池化降低分辨率
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),          # 1x1卷积
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1), # 3x3卷积
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))      # 输出: 192@12x12

    # 模块b3：第一个Inception块组
    # 包含 2个Inception模块 和 池化层
    # 通道数逐步增加(192→256→480)
    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),   # 输出通道: 64+128+32+32=256
                       Inception(256, 128, (128, 192), (32, 96), 64), # 输出通道: 128+192+96+64=480
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 输出: 480@6x6

    # 模块b4：第二个Inception块组
    # 包含 5个Inception模块 和 池化层
    # 通道数先保持(480→512)后增加(512→528→832)
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),    # 输出: 192+208+48+64=512
                       Inception(512, 160, (112, 224), (24, 64), 64),   # 输出: 160+224+64+64=512
                       Inception(512, 128, (128, 256), (24, 64), 64),   # 输出: 112+288+64+64=528
                       Inception(512, 112, (144, 288), (32, 64), 64),   # 输出: 112+288+64+64=528
                       Inception(528, 256, (160, 320), (32, 128), 128), # 输出: 256+320+128+128=832
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 输出: 832@3x3

    # 模块b5：最终分类部分
    # 包含2个Inception模块
    # 通过全局平均池化将特征图降为1x1
    # 最后展平为向量供全连接层使用
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),  # 输出: 256+320+128+128=832
                       Inception(832, 384, (192, 384), (48, 128), 128),  # 输出: 384+384+128+128=1024
                       nn.AdaptiveAvgPool2d((1,1)),   # 全局平均池化 → 1024@1x1 ，用于替代全连接层，减少参数量
                       nn.Flatten())                  # 展平为1024维向量

    # 完整网络 最终输出层：1024维 → 10类(Fashion-MNIST)
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

    X = torch.rand(size=(1, 1, 96, 96)) # 输入96x96图像，最终输出(1,10)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.1, 10, 128 # 学习率，训练轮数，(批量大小)每轮训练图像张数
    train_iter, test_iter = common.load_data_fashion_mnist(batch_size, resize=96)
    common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu())

# learn_GoogLeNet()


def learn_batchNorm():
    '''
    X: 输入数据，
        可以是 全连接层的2D张量 (batch_size, features)
        或      卷积层的4D张量 (batch_size, channels, height, width)
    gamma      : 缩放参数(可学习)
    beta       : 平移参数(可学习)
    moving_mean: 移动平均均值(用于推理阶段)
    moving_var : 移动平均方差(用于推理阶段)
    eps        ：防止除零的小常数
    momentum   ：移动平均的动量参数，控制移动平均的更新速度（默认 0.9）
    '''
    def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
        # 通过is_grad_enabled来判断当前模式是训练模式(需要计算梯度)还是预测模式(不需要梯度)
        if not torch.is_grad_enabled():
            # 预测模式：直接使用传入的移动平均所得的均值moving_mean和方差moving_var(避免训练时计算的统计量干扰)
            # 公式: (X - μ) / √(σ² + ε)，规范化输入(减去均值后除以标准差)
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            # 训练模式：计算当前批次的均值和方差
            assert len(X.shape) in (2, 4) # 检查输入维度（确保输入是2D(全连接)或4D(卷积)张量）
            if len(X.shape) == 2:
                # 使用全连接层的情况，沿样本batch维度（dim=0）计算特征维上的均值和方差
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0) # 方差
            else:
                # 卷积层：沿（batch, height, width）维度计算均值和方差（保持通道维度）
                # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
                # keepdim=True保持维度，即 保持X的形状以便后面可以做广播运算
                # 结果形状为(1, channels, 1, 1)
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

            # 标准化当前批次数据
            # 训练模式下，用当前的均值和方差做标准化
            X_hat = (X - mean) / torch.sqrt(var + eps) # eps：防止除零的小常数
            # 更新移动平均的均值和方差（指数加权平均）（更新全局统计量）
            # 公式: new_value = momentum * old_value + (1 - momentum) * current_value
            # 较大的momentum值(接近1)会给历史值更大权重
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = gamma * X_hat + beta  # 应用比例系数和比例偏移，缩放（gamma）和移位（gamma）
        return Y, moving_mean.data, moving_var.data  # 返回结果和更新后的移动平均值(使用.data获取不包含梯度的张量)

    class BatchNorm(nn.Module):
        # num_features：完全连接层的输出数量 或 卷积层的输出通道数
        # num_dims：2表示完全连接层，4表示卷积层（区分全连接层（2D）和卷积层（4D））
        def __init__(self, num_features, num_dims):
            super().__init__()
            # 根据输入类型（全连接层或卷积层）初始化形状
            if num_dims == 2:
                shape = (1, num_features) # 全连接层：形状为(1, num_features)
            else:
                shape = (1, num_features, 1, 1) # 卷积层：形状为(1, num_channels, 1, 1)

            # 可训练参数：gamma（缩放）和 beta（偏移）
            # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
            self.gamma = nn.Parameter(torch.ones(shape)) # gamma初始化为1（保持初始尺度不变）
            self.beta = nn.Parameter(torch.zeros(shape)) # beta初始化为0（保持初始偏移不变）

            # 非训练参数：移动平均的均值和方差（初始化为0和1）
            # 非模型参数的变量初始化为0和1（避免初始标准化干扰）
            self.moving_mean = torch.zeros(shape)
            self.moving_var = torch.ones(shape)

        def forward(self, X):
            # 确保移动平均统计量与输入数据在同一设备上（CPU/GPU）
            # 如果X不在内存上，将moving_mean和moving_var
            # 复制到X所在显存上
            if self.moving_mean.device != X.device:
                self.moving_mean = self.moving_mean.to(X.device)
                self.moving_var = self.moving_var.to(X.device)

            # 调用底层批归一化函数，并更新移动平均值
            # 保存更新过的moving_mean和moving_var
            Y, self.moving_mean, self.moving_var = batch_norm(
                X, self.gamma, self.beta, self.moving_mean,
                self.moving_var, eps=1e-5, momentum=0.9)
            return Y

    net = nn.Sequential(
        # 卷积层 + BatchNorm + 激活函数
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Flatten(),

        # 全连接层 + BatchNorm + 激活函数
        nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
        nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
        nn.Linear(84, 10)) # 输出层（无 BatchNorm 和激活函数）

    lr, num_epochs, batch_size = 1.0, 10, 256 # 学习率，训练轮数，每轮处理的批量大小
    train_iter, test_iter = common.load_data_fashion_mnist(batch_size) # 加载数据集
    # common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu()) # 训练

    print(f"从第一个批量规范化层中学到的\n"
          f"拉伸参数 gamma：\n{net[1].gamma.reshape((-1,))},\n"
          f"偏移参数 beta： \n{net[1].beta.reshape((-1,))}")

    # 直接使用pytorch框架中的BatchNorm
    net_useAPI = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
        nn.Linear(84, 10))
    common.train_ch6(net_useAPI, train_iter, test_iter, num_epochs, lr, common.try_gpu())
# learn_batchNorm()




