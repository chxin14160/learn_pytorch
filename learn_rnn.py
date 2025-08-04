import torch
from torch import nn
import common
import collections # 提供高性能的容器数据类型，替代Python的通用容器(如 dict, list, set, tuple)
import re # 供正则表达式支持，用于字符串匹配、搜索和替换
import random
import math
from torch.nn import functional as F


def learn_SequenceModel():
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

    # 简单的K步预测：使用预测 来进行K步预测（递归预测）
    # 是严格的递归预测，每个新预测都基于之前的预测
    # 潜在问题：递归预测的误差会累积，因为每个预测都基于之前的预测
    multistep_preds = torch.zeros(T) # 初始化预测结果张量
    multistep_preds[: n_train + tau] = x[: n_train + tau] # 用真实值填充前面已知的真实值
    for i in range(n_train + tau, T): # 递归预测
        # 使用前tau个预测值作为输入，预测下一个值
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1)))

    common.plot([time, time[tau:], time[n_train + tau:]],
             [x.detach().numpy(), onestep_preds.detach().numpy(),
              multistep_preds[n_train + tau:].detach().numpy()], 'time',
             'x', legend=['data', '1-step preds', 'multistep preds'],
             xlim=[1, 1000], figsize=(6, 3))


    # 多步预测（序列预测）
    # 是序列预测，可以同时获得多个未来时间步的预测（虽然这些中间预测也基于之前的预测）
    # 潜在问题：虽然能一次预测多个步长，但长期预测仍然依赖中间预测结果
    max_steps = 64 # 最大预测步数

    # 初始化特征张量，(要预测的样本数,特征数),其中
    # 前 tau 列：存储真实历史数据（作为输入）
    # 后 max_steps 列：存储模型预测的未来值
    # T-tau-max_steps+1是可计算的时间窗口数量，特征数(tau列真实数据 + max_steps列预测数据)
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))

    # 前tau列用真实值填充
    # 列i（i<tau）是来自x的观测(实际真实值)，其时间步从（i）到（i+T-tau-max_steps+1）
    print(f"真实值填充：{x[i: i + T - tau - max_steps + 1].shape}") # torch.Size([1])
    for i in range(tau):
        features[:, i] = x[i: i + T - tau - max_steps + 1]
        # 对于 i=0，features[:, 0] = x[0 : 0 + 927]（即 x[0] 到 x[926]）
        # 对于 i=1，features[:, 1] = x[1 : 1 + 927]（即 x[1] 到 x[927]）
        # ...
        # 对于 i=9，features[:, 9] = x[9 : 9 + 927]（即 x[9] 到 x[935]）

    # 后max_steps列用模型预测填充
    # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1) # .reshape(-1)展平为一维向量

    steps = (1, 4, 16, 64)  # 要展示的预测步数
    common.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
             [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
             legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
             figsize=(6, 3))
# learn_SequenceModel()


# 下载器与数据集配置
# 为 time_machine 数据集注册下载信息，包括文件路径和校验哈希值（用于验证文件完整性）
downloader = common.C_Downloader()
DATA_HUB = downloader.DATA_HUB  # 字典，存储数据集名称与下载信息
DATA_URL = downloader.DATA_URL  # 基础URL，指向数据集的存储位置

DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


def learn_textPreprocess():
    # # 加载文本数据
    # def read_time_machine():  #@save
    #     """将时间机器数据集加载到文本行的列表中"""
    #     # 通过 downloader.download('time_machine') 获取文件路径
    #     with open(downloader.download('time_machine'), 'r') as f:
    #         lines = f.readlines() # 逐行读取文本文件
    #     # 用正则表达式 [^A-Za-z]+ 替换所有非字母字符为空格
    #     # 调用 strip() 去除首尾空格，lower() 转换为小写
    #     # 返回值：处理后的文本行列表（每行是纯字母组成的字符串）
    #     return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

    lines = common.read_time_machine(downloader)
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])     # 第1行内容
    print(lines[10])    # 第11行内容

    # # 词元化函数：支持按单词或字符拆分文本
    # # lines：预处理后的文本行列表
    # # token：词元类型，可选 'word'（默认）或 'char
    # # 返回值：嵌套列表，每行对应一个词元列表
    # def tokenize(lines, token='word'):  #@save
    #     """将文本行拆分为单词或字符词元"""
    #     if token == 'word':
    #         return [line.split() for line in lines]  # 按空格分词
    #     elif token == 'char':
    #         return [list(line) for line in lines]   # 按字符拆分
    #     else:
    #         print('错误：未知词元类型：' + token)

    tokens = common.tokenize(lines)
    for i in range(11):
        print(f"第{i}行：{tokens[i]}")

    '''
    假设原始文本前两行为：
    The Time Machine, by H. G. Wells [1898]
    I
    预处理后：['the time machine by h g wells', 'i']
    词元化结果：[['the', 'time', 'machine', 'by', 'h', 'g', 'wells'], ['i']]
    '''

    vocab = common.Vocab(tokens) # 构建词表，管理词元与索引的映射关系
    print(f"前几个高频词及其索引：\n{list(vocab.token_to_idx.items())[:10]}")

    for i in [0, 10]: # 将每一条文本行转换成一个数字索引列表
        print(f"第{i}行信息：")
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])


    # # 获取《时光机器》的 词元索引序列和词表对象
    # # max_tokens：限制返回的词元索引序列的最大长度（默认 -1 表示不限制）
    # def load_corpus_time_machine(max_tokens=-1):  #@save
    #     """返回时光机器数据集的词元索引列表和词表"""
    #     lines = read_time_machine() # 加载文本数据，得到文本行列表
    #     tokens = tokenize(lines, 'char') # 词元化：文本行列表→词元列表，按字符级拆分
    #     vocab = common.Vocab(tokens) # 构建词表
    #     # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    #     # 所以将所有文本行展平到一个列表中
    #     # vocab[token] 查询词元的索引（若词元不存在，则返回0，即未知词索引）
    #     # corpus：list，每个元素为词元的对应索引
    #     corpus = [vocab[token] for line in tokens for token in line] # 展平词元并转换为索引
    #     if max_tokens > 0: # 限制词元序列长度
    #         corpus = corpus[:max_tokens] # 截断 corpus 到前 max_tokens 个词元
    #     # corpus：词元索引列表（如 [1, 2, 3, ...]）
    #     # vocab：Vocab对象，用于管理词元与索引的映射
    #     return corpus, vocab

    corpus, vocab = common.load_corpus_time_machine(downloader) # 加载数据
    print(f"corpus词元索引列表的长度：{len(corpus)}")
    print(f"词表大小：{len(vocab)}")
    print(f"词频统计（降序）：\n{vocab.token_freqs}")
    # 索引 ↔ 词元转换
    print(f"前10个索引对应的词元：\n{vocab.to_tokens(corpus[:10])}")
    print(f"前10个词元对应的索引：\n{corpus[:10]}")
    print(f"前10个词元对应的索引：\n{[idx for idx in corpus[:10]]}")
# learn_textPreprocess()


def learn_languageModelsAndDatasets():
    lines = common.read_time_machine(downloader) # 获取文本行列表
    tokens = common.tokenize(lines) # 将文本行列表中的元素词元化(按单词拆分)
    # 因为每个文本行不一定是一个句子或一个段落，因此把所有文本行拼接到一起
    corpus = [token for line in tokens for token in line] # 将词元列表展平
    vocab = common.Vocab(corpus)
    print(f"前10个最常用的（频率最高的）单词：\n{vocab.token_freqs[:10]}")

    freqs = [freq for token, freq in vocab.token_freqs] # 词频(降序)
    common.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
             xscale='log', yscale='log') # 绘制(横坐标=词频索引，纵坐标=词频具体数值)

    # 词元组合(二元语法)
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = common.Vocab(bigram_tokens)
    print(f"前10个最常用的（频率最高的）词元组合(二元语法)：\n{bigram_vocab.token_freqs[:10]}")

    # 词元组合(三元语法)
    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = common.Vocab(trigram_tokens)
    print(f"前10个最常用的（频率最高的）词元组合(三元语法)：\n{trigram_vocab.token_freqs[:10]}")

    # 再直观对比 三种模型中的词元频率：一元语法、二元语法和三元语法
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    common.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
             ylabel='frequency: n(x)', xscale='log', yscale='log',
             legend=['unigram', 'bigram', 'trigram'])


    my_seq = list(range(35)) # 生成一个从0到34的序列
    # 批量大小为2，时间步数为5
    for idx, (X, Y) in enumerate(common.seq_data_iter_random(my_seq, batch_size=2, num_steps=5)):
        print(f" 随机取样 —————— idx={idx} —————— \n"
              f"X: {X}\nY:{Y}")

    for idx, (X, Y) in enumerate(common.seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5)):
        print(f" 顺序分区 —————— idx={idx} —————— \n"
              f"X: {X}\nY:{Y}")
# learn_languageModelsAndDatasets()


# 验证：分别矩阵乘法后再结果相加 相当于 输入和权重分别拼接后再矩阵乘法
def learn_422():
    # （1）分别矩阵乘法后 再结果相加
    # 首先定义矩阵 X、W_xh、H 和 W_hh，分别为(3,1)、(1,4)、(3,4)和(4,4)
    # .normal() 从离散正态分布(均值为0，标准差为1)中抽取随机数
    X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
    H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
    # 分别将X乘以W_xh，将H乘以W_hh，然后将这两个乘法相加
    temp_add = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
    print(f"得到一个形状为(3,4)的矩阵：\n{temp_add}")

    # （2）输入和权重分别拼接后 再矩阵乘法
    # .cat((X, H), 1) 沿列（轴1）拼接矩阵X和H
    # .cat((W_xh, W_hh), 0)) 沿行（轴0）拼接矩阵W_xh和W_hh
    # 这两个拼接分别产生形状(3,5)和形状(5,4)的矩阵
    all_input = torch.cat((X, H), 1)
    all_w = torch.cat((W_xh, W_hh), 0)
    print(f"拼接后的输入：\n{all_input}")
    print(f"拼接后的权重：\n{all_w}")
    temp_add = torch.matmul(all_input, all_w) # 再将这两个拼接的矩阵相乘
    print(f"得到与上面相同形状(3,4)的输出矩阵：\n{temp_add}")
# def learn_422()



batch_size, num_steps = 32, 35 # 每个小批量包含32个子序列，每个子序列的词元数为35
train_iter, vocab = common.load_data_time_machine(downloader, batch_size, num_steps) # 词表对象

# 将索引 [0, 2] 转换为长度为 len(vocab) 的 one-hot 编码，
# 结果是一个形状为 (2, len(vocab)) 的张量，其中每一行是对应索引的 one-hot 向量
F.one_hot(torch.tensor([0, 2]), len(vocab))
print(f"索引为0和2的独热向量：\n{F.one_hot(torch.tensor([0, 2]), len(vocab))}")

X = torch.arange(10).reshape((2, 5)) # 创建形状为 (2, 5) 的张量 X，包含 0~9 的整数
# X.T 转置X，得到形状 (5, 2)，表示5个时间步（或序列长度），每个时间步有2个特征（或2个独立的索引）
# one_hot(X.T, 28) 对X.T应用one_hot编码，num_classes=28 (5个时间步 × 2个索引 × 每个索引的28维one-hot编码)
print(f"{F.one_hot(X.T, 28).shape}")

"""
初始化RNN模型的参数
参数:
vocab_size: 词表大小 (输入和输出的维度)
num_hiddens: 隐藏层大小
device: 计算设备 (CPU/GPU)
返回:params: 模型参数列表 [W_xh, W_hh, b_h, W_hq, b_q]
"""
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size # 输入和输出的维度都是词表大小

    def normal(shape):  # 定义正态分布初始化函数
        return torch.randn(size=shape, device=device) * 0.01 # 生成小随机数

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))        # 输入到隐藏层的权重
    W_hh = normal((num_hiddens, num_hiddens))       # 隐藏层到隐藏层的权重
    b_h = torch.zeros(num_hiddens, device=device)   # 隐藏层偏置
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))       # 隐藏层到输出层的权重
    b_q = torch.zeros(num_outputs, device=device)   # 输出层偏置
    # 附加梯度 (将所有参数放入列表，并设置需要梯度)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)  # 开启梯度追踪
    return params

'''
初始化RNN的隐藏状态：初始化RNN模型的参数
包括输入到隐藏层、隐藏层到隐藏层、隐藏层到输出层的权重以及相应的偏置
权重使用小随机数初始化，偏置初始化为零
batch_size: 批量大小
num_hiddens: 隐藏层大小
device: 计算设备
返回:包含初始隐藏状态的元组 (H0,)
'''
def init_rnn_state(batch_size, num_hiddens, device):
    # 创建全零的初始隐藏状态，形状为 (batch_size, num_hiddens)
    return (torch.zeros((batch_size, num_hiddens), device=device), )  # 初始隐藏状态为全0

'''
RNN前向传播函数
功能：返回初始的隐藏状态（一个全零张量），形状为 (batch_size, num_hiddens)
inputs: 输入序列，形状为 (时间步数量, 批量大小, 词表大小)
state: 初始隐藏状态
params: 模型参数
返回:
outputs: 所有时间步的输出，形状为 (时间步数量 * 批量大小, 词表大小)
new_state: 新的隐藏状态
'''
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params # 解包参数
    H, = state                          # 解包隐藏状态
    outputs = []                        # 用于存储每个时间步的输出
    # X的形状：(批量大小，词表大小)
    for X in inputs:# 遍历每个时间步的输入
        # 计算新的隐藏状态：H = tanh(X * W_xh + H * W_hh + b_h)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q # 计算当前时间步的输出：Y = H * W_hq + b_q
        outputs.append(Y)  # 保存输出
    # 将所有时间步的输出沿时间维度拼接
    return torch.cat(outputs, dim=0), (H,) # 返回输出和最后一个隐藏状态

'''
RNN模型类
功能：实现RNN的前向传播
对输入序列的每个时间步，计算隐藏状态和输出
输入inputs形状为 (时间步数, 批量大小, 词表大小)（实际在调用前会转成one-hot）
该函数返回所有时间步的输出（拼接成一个张量）和最后一个时间步的隐藏状态
'''
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens # 词表大小,隐藏层大小
        self.params = get_params(vocab_size, num_hiddens, device)   # 初始化参数
        self.init_state, self.forward_fn = init_state, forward_fn   # 初始化隐藏状态的函数,前向传播函数

    """
    模型调用方法（前向传播）
    参数:
    X: 输入序列，形状为 (批量大小, 时间步数量)
    state: 隐藏状态
    返回:输出和新的隐藏状态
    """
    def __call__(self, X, state):
        # 将输入X转换为one-hot编码
        # X.T: 转置为 (时间步数量, 批量大小)
        # one_hot: 转换为 (时间步数量, 批量大小, 词表大小)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params) # 调用前向传播函数

    def begin_state(self, batch_size, device):
        """获取初始隐藏状态"""
        return self.init_state(batch_size, self.num_hiddens, device) # 返回初始隐藏状态

num_hiddens = 512 # 隐藏层大小
net = RNNModelScratch(len(vocab), num_hiddens, common.try_gpu(), get_params,
                      init_rnn_state, rnn) # 创建模型实例
state = net.begin_state(X.shape[0], common.try_gpu()) # 获取初始状态（假设X是一个批量的输入数据）
Y, new_state = net(X.to(common.try_gpu()), state) # 将数据X转移到设备（如GPU）并前向传播
print(f"输出Y的形状：{Y.shape}") # (时间步数量 * 批量大小, 词表大小)
print(f"隐藏状态的元组长度：{len(new_state)}")  # 隐藏状态的元组长度: 1
print(f"隐藏状态的形状：{new_state[0].shape}") # (批量大小, 隐藏层大小)






