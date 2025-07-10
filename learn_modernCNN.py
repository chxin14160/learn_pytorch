import torch
from torch import nn
import common


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

# 模拟输入：batch_size=1，通道=1，高度=224，宽度=224（为了能实用AlexNet架构）
X = torch.randn(1, 1, 224, 224)
for layer in net:   # 逐层前向传播并打印输出形状
    X=layer(X)      # 前向计算
    print(layer.__class__.__name__,'output shape:\t\t',X.shape)

batch_size = 128 # 每次处理128张图片
# 加载Fashion-MNIST数据集，返回 训练集和测试集的DataLoader对象
train_iter, test_iter = common.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10 # 学习率0.01，训练10轮
common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu())




