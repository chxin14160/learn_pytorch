import torch
from torch import nn
import common



# 1. 数据生成及可视化
# 生成含噪声的周期性时间序列数据（正弦波+噪声）
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)      # 时间步 [1, 2, ..., 1000]
# (T,) 是表示张量形状（shape）的元组，用于指定生成的高斯噪声(正态分布)的维度（指定生成一维张量，长度为T）
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,)) # 生成正弦信号 + 高斯噪声
print(f"x的形状：{x.shape}")
common.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3)) # 绘制时间序列

# 2. 构造特征与标签
# 将时间序列转换为监督学习问题（用前4个点预测第5个点
tau = 4 # 用过去4个时间步预测下一个时间步
features = torch.zeros((T - tau, tau)) # 特征矩阵形状: (996, 4)（总共996个有效样本，每个样本对应4个特征）
for i in range(tau):
    features[:, i] = x[i: T - tau + i] # 滑动窗口填充特征
labels = x[tau:].reshape((-1, 1))      # 标签形状: (996, 1) （前4项丢弃）

# 3. 数据加载器
# 创建数据迭代器，支持批量训练
batch_size, n_train = 16, 600 # 批量大小16，训练集600样本
# 将前n_train个样本用于训练
train_iter = common.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True) # 创建数据迭代器，支持批量训练

# 4. 网络初始化
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) # Xavier初始化权重

# 定义一个简单的多层感知机（MLP）
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),  # 输入层(4) → 隐藏层(10)
                        nn.ReLU(),         # 激活函数
                        nn.Linear(10, 1))  # 隐藏层(10) → 输出层(1)
    net.apply(init_weights)  # 应用初始化
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
# reduction='none' 返回每个样本的损失，后续需手动 .sum() 或 .mean()
loss = nn.MSELoss(reduction='none')  # 均方误差损失，不自动求和/平均

# 对模型进行训练和测试
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = common.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)             # 模型预测输出结果
        y = y.reshape(out.shape) # 将实际标签y的形状调整为与模型输出out一致
        l = loss(out, y)         # 模型输出out与实际标签y之间的损失
        metric.add(l.sum(), l.numel()) # 将损失总和 和 样本总数 累加到metric中
    return metric[0] / metric[1] # 损失总和/预测总数，即平均损失

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)  # Adam优化器
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()       # 梯度清零
            l = loss(net(X), y)       # 计算损失（形状[batch_size, 1]）
            l.sum().backward()        # 反向传播（对所有样本损失求和）
            trainer.step()            # 更新参数
        # 打印训练损失（假设evaluate_loss是自定义函数）
        print(f'epoch {epoch + 1}, '
              f'loss: {evaluate_loss(net, train_iter, loss):f}')

net = get_net()      # 初始化网络
train(net, train_iter, loss, 5, 0.01)  # 训练5个epoch，学习率0.01


# 单步预测：模型预测下一时间步的能力
onestep_preds = net(features)
common.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

# K步预测：使用预测 来进行K步预测
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

common.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))


max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
common.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))











