import torch
from torch import nn
from torch.nn import functional as F


def layers_and_blocks():
    net = nn.Sequential(
        nn.Linear(20, 256),
        nn.ReLU(),
        nn.Linear(256, 10))
    X = torch.rand(2, 20)
    print(f"随机生成的原始输入：\n{X}")
    print(f"模型输出：\n{net(X)}")

    class MLP(nn.Module):
        # 用模型参数声明层，这里声明两个全连接的层
        def __init__(self):
            # 调用MLP的父类Module的构造函数来执行必要的初始化
            # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
            super().__init__()
            self.hidden = nn.Linear(20, 256)  # 隐藏层
            self.out = nn.Linear(256, 10)  # 输出层

        # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
        def forward(self, X):
            # 注意：这里使用ReLU的函数版本，其在nn.functional模块中定义。
            return self.out(F.relu(self.hidden(X)))
    net = MLP()
    mlp_Out = net(X)
    print(f"MLP模型输出：\n{mlp_Out}")

    # 直接将神经网络的每个层当作参数传进来
    class MySequential(nn.Module):
        def __init__(self, *args):
            super().__init__() # 调用父类的构造函数
            for idx, module in enumerate(args):
                # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
                # 变量_modules中。_module的类型是OrderedDict
                self._modules[str(idx)] = module

        def forward(self, X):
            # OrderedDict保证了按照成员添加的顺序遍历它们
            for block in self._modules.values():
                X = block(X)
            return X
    net = MySequential(
        nn.Linear(20, 256),
        nn.ReLU(),
        nn.Linear(256, 10))
    myModule_Out = net(X)
    print(f"MySequential模型输出：\n{myModule_Out}")

    class FixedHiddenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            # 不计算梯度的随机权重参数。因此其在训练期间保持不变
            self.rand_weight = torch.rand((20, 20), requires_grad=False)
            self.linear = nn.Linear(20, 20)

        def forward(self, X):
            X = self.linear(X)
            # 使用创建的常量参数以及relu和mm函数
            X = F.relu(torch.mm(X, self.rand_weight) + 1)
            # 复用全连接层。这相当于两个全连接层共享参数
            X = self.linear(X)
            # 控制流
            while X.abs().sum() > 1:
                X /= 2
            return X.sum()

    net = FixedHiddenMLP()
    fixMLP_Out = net(X)
    print(f"FixedHiddenMLP模型输出：\n{fixMLP_Out}")

    class NestMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU())
            self.linear = nn.Linear(32, 16)

        def forward(self, X):
            return self.linear(self.net(X))
    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    print(f"NestMLP模型输出：\n{chimera(X)}")

# layers_and_blocks()


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(f"随机生成的原始输入：\n{X}")
print(f"模型输出：\n{net(X)}")

print(f"第二个全连接层的参数:\n{net[2].state_dict()}")

# 从第二个全连接层（即第三个神经网络层）提取偏置:
# 参数是复合的对象，包含值、梯度和额外信息，因此需要显式参数值
print(f"类型：{type(net[2].bias)}")
print(f"值(包括其形状和数据类型)：\n{net[2].bias}")
print(f"数据部分(偏置参数的底层数据张量)：{net[2].bias.data}")

print(f"参数的梯度：{net[2].weight.grad == None}")


print(f"---访问第一个全连接层的参数：\n{[(name, param.shape) for name, param in net[0].named_parameters()]}")
print(f"---访问所有层：\n{[(name, param.shape) for name, param in net.named_parameters()]}")
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# net.state_dict() 返回模型的参数字典
# ['2.bias'] 从参数字典中 获取模型中第三个模块(索引从0开始)的偏置参数
print(f"另一种访问网络参数的方式：{net.state_dict()['2.bias'].data}")

'''从嵌套块收集参数'''
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(f"嵌套块 模型输出：\n{rgnet(X)}")

print(f"嵌套块 模型结构：\n{rgnet}")
print(f"第一个主要的块中、第二个子块的第一层的偏置项：\n{rgnet[0][1][0].bias.data}")


'''参数初始化'''
# 将所有权重参数初始化为 标准差=0.01的高斯随机变量，且将偏置参数设置为0
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(f"权重初始化为 标准差=0.01的高斯随机变量：\n{net[0].weight.data[0]}")
print(f"偏置初始化为 常量0：\n{net[0].bias.data[0]}")

# 将所有参数初始化为给定的常数，比如初始化为1
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(f"权重初始化为 常量1：\n{net[0].weight.data[0]}")
print(f"偏置初始化为 常量0：\n{net[0].bias.data[0]}")

'''
Xavier初始化：
    根据输入和输出的维度自动调整初始化的范围，
    使得每一层的输出的方差在训练初期保持一致。
    （助于缓解梯度消失和梯度爆炸）
均匀分布：
    xavier_uniform_()使用均匀分布来初始化权重，而不是正态分布。
    均匀分布的范围是[-limit, limit]，其中limit是根据输入和输出维度计算得出的
'''
# 对某些块应用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)   # 使用Xavier初始化方法初始化第一个神经网络层
net[2].apply(init_42)       # 将第三个神经网络层初始化为常量值42
print(f"第一层网络，使用Xavier初始化权重：\n{net[0].weight.data[0]}")
print(f"第三层网络，权重初始化为 常量42：\n{net[2].weight.data}")


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10) # 使用均匀分布初始化权重，范围是[-10, 10]
        # 对权重进行阈值处理，将绝对值小于 5 的元素设置为 0。
        # 即 只有 绝对值>=5 的权重会被保留，其余的被置零
        m.weight.data *= m.weight.data.abs() >= 5
net.apply(my_init)
print(f"自定义初始化：\n{net[0].weight[:2]}")

net[0].weight.data[:] += 1      # 将第一个模块的所有权重值加 1
net[0].weight.data[0, 0] = 42   # 将第一个模块的权重张量的 [0, 0] 位置的值设置为 42
print(f"对权重进行手动修改后：\n{net[0].weight[:2]}")





