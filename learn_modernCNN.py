import torch
from torch import nn
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









