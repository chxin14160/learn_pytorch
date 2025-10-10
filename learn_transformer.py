import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import common


# 下载器与数据集配置
# 为 time_machine 数据集注册下载信息，包括文件路径和校验哈希值（用于验证文件完整性）
downloader = common.C_Downloader()
DATA_HUB = downloader.DATA_HUB  # 字典，存储数据集名称与下载信息
DATA_URL = downloader.DATA_URL  # 基础URL，指向数据集的存储位置

# 注册数据集信息到DATA_HUB全局字典
# 格式：(数据集URL, MD5校验值)
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip', # 完整下载URL（DATA_URL是d2l定义的基准URL）
                           '94646ad1522d915e7b0f9296181140edcf86a4f5') # 文件MD5，用于校验下载完整性


# print(f"10x10 的单位矩阵（对角线为 1，其余为 0）：\n{torch.eye(10)}")

# 仅当查询和键相同时，注意力权重为1，否则为0
# torch.eye(10): 生成一个 10x10 的单位矩阵（对角线为 1，其余为 0）
# .reshape((1,1,10,10)): 调整形状为(batch_size=1,num_heads=1,seq_len=10,seq_len=10)
# 模拟一个单头注意力机制的权重矩阵（Queries × Keys）
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
# common.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')



n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5) # 排序后的训练样本(以便可视化)

def f(x): # 非线性函数
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出(添加噪声项)
x_test = torch.arange(0, 5, 0.1) # 测试样本
y_truth = f(x_test)              # 测试样本的真实输出
n_test = len(x_test)             # 测试样本数
print(f"测试样本数：{n_test}")

def plot_kernel_reg(y_hat):
    common.plot_kernel_reg(x_test, y_truth, y_hat, x_train, y_train)

def average_aggregation():
    '''👉 平均汇聚'''
    # .repeat_interleave() 用于重复张量元素
    # 计算出训练样本标签y的均值，然后将其重复n_test次，返回一个长度为n_test的 1D张量
    # 即，生成一个长度为 n_test 的张量，所有元素为 y_train的均值，即 y_train.mean()
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    # plot_kernel_reg(y_hat)
    common.plot_kernel_reg(x_test, y_truth, y_hat, x_train, y_train)
# average_aggregation()

def nonParametric_attention_aggregation():
    '''👉 非参数注意力汇聚'''
    # 1. 重复x_test以匹配注意力权重的形状
    # 为每个测试样本生成一个与 x_train 对齐的矩阵，便于后续计算相似度
    # X_repeat的形状:(n_test,n_train),
    # 每一行都包含着相同的测试输入（例如：同样的查询）
    # x_test.repeat_interleave(n_train) 对张量x_test的每个元素沿指定维度(默认0)重复n_train次
    # 即 每个测试样本重复 n_train 次（展平为一维）
    # .reshape((-1, n_train)) 将展平的张量重新调整为(n_test, n_train)，其中每一行是x_test的一个副本 重复n_train次
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    print(f"测试数据形状x_test.shape：{x_test.shape}")
    print(f"将测试数据元素重复后形状 x_test.repeat_interleave(n_train).shape："
          f"{x_test.repeat_interleave(n_train).shape}")
    print(f"重塑形状为(n_test,n_train)以便后续计算相似度（相似度越高，注意力权重越大）\n"
          f"x_test.repeat_interleave(n_train).reshape((-1, n_train)).shape："
          f"{x_test.repeat_interleave(n_train).reshape((-1, n_train)).shape}")

    # 2. 计算注意力权重（高斯核软注意力）
    # x_train包含着键。attention_weights的形状：(n_test,n_train),
    # 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
    # -(X_repeat - x_train)**2 / 2 计算测试样本与训练样本之间的负欧氏距离（高斯核的指数部分）
    # 测试数据的输入X_repeat 相当于 查询
    # 训练数据的输入x_train  相当于 键
    attention_weights = nn.functional.softmax(
        -(X_repeat - x_train)**2 / 2, # 高斯核(负欧氏距离)，相似度计算：距离越小，相似度越高（权重越大）
        dim=1) # 对每一行（每个测试样本）做softmax归一化，确保权重和为1

    # 3. 加权平均得到预测值
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
    # 左：每个测试样本 对应所有训练标签的各个注意力权重
    # 右：每个训练样本对应的标签
    y_hat = torch.matmul(attention_weights, y_train) # 矩阵乘法：左找行，右找列
    plot_kernel_reg(y_hat)

    # np.expand_dims(attention_weights, 0) 在第0轴(最外层)扩展新维度
    # np.expand_dims(np.expand_dims(attention_weights,0),0) 连续两次在第0轴(最外层)扩展新维度
    # 假设attention_weights原始维度是(3,4)，则第一次扩展变成(1,3,4)，则第一次扩展变成(1,1,3,4)
    common.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs') # 显示注意力权重的矩阵热图
# nonParametric_attention_aggregation()

def parametric_attention_aggregation():
    ''' 👉 带参数注意力汇聚 '''
    X = torch.ones((2, 1, 4))
    Y = torch.ones((2, 4, 6))
    print(f"批量矩阵乘法bmm后，结果矩阵形状：{torch.bmm(X, Y).shape}")

    # 演示注意力机制中的加权求和
    weights = torch.ones((2, 10)) * 0.1          # 形状: (批量大小, 序列长度)-注意力权重
    values = torch.arange(20.0).reshape((2, 10)) # 形状: (批量大小, 序列长度)-值向量
    # .unsqueeze()在指定位置增加维度
    print(f"在指定位置增加维度后矩阵形状：\n"
          f"weights：{weights.unsqueeze(1).shape}\n"
          f"values ：{values.unsqueeze(-1).shape}")
    print(f"增加维度后 进行 批量矩阵乘法bmm，结果为：\n"
          f"使用bmm计算加权和: 权重(2,1,10) × 值(2,10,1) = 结果(2,1,1) \n"
          f"{torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))}")


    class NWKernelRegression(nn.Module):
        ''' Nadaraya-Watson 核回归模型,实现基于注意力机制的核回归
        实现Nadaraya-Watson核回归的非参数方法，通过注意力机制对输入数据进行加权平均
        使用高斯核函数来计算查询与键之间的相似度，并将这些相似度作为权重对值进行加权求和
        '''
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # 可学习的参数 (高斯核的带宽参数)，即 查询与键间距离要乘以的权重
            self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

        def forward(self, queries, keys, values):
            '''
            queries : 查询输入 (n_query,)，有n_query个查询
            keys    : 训练输入 (n_query, n_train)，即 每个查询对应n_train个键
            values  : 训练输出 (n_query, n_train)
            # queries 和 attention_weights的形状为 (查询个数，“键－值”对个数)
            返回: 加权求和后的预测值 (n_query,)
            '''
            # 扩展 查询向量queries形状 以匹配 键值对keys的维度
            # queries形状: (查询个数,) -> 扩展为 (查询个数, 键值对个数)
            # 将查询的每个元素重复 键的列数次（为了当前查询与每个键做差）
            # 然后将查询的形状重塑为 列维与键的行维相等的 矩阵形式 (第i行元素皆为第i个查询)
            # 由此可使查询拥有 与键相同的形状，以便后续做差计算
            queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))

            # 计算注意力权重（使用高斯核函数）
            # 公式: attention = softmax(-(query - key)^2 * w^2 / 2)
            # 注意力权重通过高斯核 exp(-(x_query-x_key)^2 / (2σ^2)) 计算，
            # 使用softmax进行归一化，确保所有权重之和为1
            self.attention_weights = nn.functional.softmax(
                -((queries - keys) * self.w) ** 2 / 2, dim=1) # 形状 (n_query, n_train)

            # 使用注意力权重对值进行加权求和得到预测值
            # bmm: (批量大小, 1, 键值对个数) × (批量大小, 键值对个数, 1) = (批量大小, 1, 1)
            # values的形状为(查询个数，“键－值”对个数)
            # 注意力权重：在除批次维以外的第一维增加一维；
            # 值：在最后一维增加一个维度
            # 然后各自增维后的注意力权重与值 执行批量矩阵乘法，计算完成后重塑形状
            return (torch.bmm(self.attention_weights.unsqueeze(1), # (n_query, 1, n_train)
                             values.unsqueeze(-1)) # (n_query, n_train, 1)
                             .reshape(-1)) # (n_query, 1, 1) → (n_query,)

    print(f"x_train.shape={x_train.shape}") # ([50])
    print(f"repeat((n_train, 1)).shape={x_train.repeat((n_train, 1)).shape}") # ([50, 50])

    # 准备训练时的 keys 和 values （用于自注意力）
    # X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
    # Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
    # x_train第0维直接重复x_train次(对应训练数据的个数)，第1维重复一次(保持不变)
    # 原本x_train是长度为50的向量形状，.repeat后形状变为(训练数据总是, 50)
    X_tile = x_train.repeat((n_train, 1)) # 形状 (n_train * n_train, dim)
    Y_tile = y_train.repeat((n_train, 1)) # 形状 (n_train * n_train, dim)

    # （排除对角线元素，即自身。避免自匹配）
    # mask 用于排除自匹配（即查询点不与自身计算注意力）
    # 1与对角线为1的单位矩阵做差，再转换为bool类型 (使对角区域为false，以便排除)
    mask = (1 - torch.eye(n_train)).type(torch.bool) # 形状 (n_train, n_train)
    # 等效于以下两种方法
    # mask = ~torch.eye(n_train, dtype=torch.bool) # 方法2：直接创建布尔掩码（更高效）
    # mask = (torch.eye(n_train) == 0) # 方法3：使用比较操作

    # 创建键和值
    # keys的形状  :('n_train'，'n_train'-1)
    # values的形状:('n_train'，'n_train'-1)
    # 通过掩码mask从 _tile中选择元素(每行元素皆少了对角线位置的那个)，然后再重新排列(行数不变)
    keys   = X_tile[mask].reshape((n_train, -1)) # 形状 (n_train, n_train-1)
    values = Y_tile[mask].reshape((n_train, -1)) # 形状 (n_train, n_train-1)


    # 初始化模型、损失函数和优化器
    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none') # 逐元素计算损失
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = common.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

    for epoch in range(5):
        trainer.zero_grad() # 清零梯度
        # 计算每个训练点的预测值（queries=x_train, keys/values来自其他点）
        l = loss(net(x_train, keys, values), y_train) # 前向传播
        l.sum().backward()  # 反向传播（对损失求和后反向传播）
        trainer.step()      # 更新参数
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
        animator.add(epoch + 1, float(l.sum()))


    # 测试阶段：每个测试点与所有训练点计算注意力
    # keys的形状 :(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
    # value的形状:(n_test，n_train)，每一行都包含相同的训练输出（例如，相同的值）
    # x_train第0维元素重复n_test次(对应测试数据的个数)，第2维元素重复一次(不变)
    # 原本x_train是长度为50的向量形状，.repeat后键和值的形状变为(测试数据总数, 50)
    keys   = x_train.repeat((n_test, 1)) # 形状 (n_test, n_train)
    values = y_train.repeat((n_test, 1)) # 形状 (n_test, n_train)
    y_hat = net(x_test, keys, values).unsqueeze(1).detach() # 预测
    plot_kernel_reg(y_hat) # 绘制回归结果

    # 可视化注意力权重（测试点 vs 训练点）
    # net.attention_weights形状: (n_test, n_train)
    # 添加两个维度使其变为(1, 1, n_test, n_train)以匹配show_heatmaps期望的4D输入
    common.show_heatmaps(
        net.attention_weights.unsqueeze(0).unsqueeze(0), # 增加批次和头维度
        xlabel='Sorted training inputs',
        ylabel='Sorted testing inputs')
# parametric_attention_aggregation()

def attention_scoring_function():
    ''' 注意力评分函数 '''
    # 演示：掩蔽softmax操作
    print(f"两个有效长度分别为2和3的 2×4矩阵，经过掩蔽softmax操作后结果：\n"
          f"{common.masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))}")
    print(f"以二维张量作为输入指定每行的有效长度：\n"
          f"{common.masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))}")

    # 创建测试数据
    # 查询、键和值的形状为（批量大小，步数或词元序列长度，特征大小）
    # 实际输出 为 q(2, 1, 20)、 k(2, 10, 2)、 v(2, 10, 4)
    # 从均值为0，标准差为1的正态分布中随机抽取值来初始化查询q，键k初始化为全0
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6]) # 两个批量的有效长度分别为2和6

    def learn_AdditiveAttention():
        ''' 加性注意力机制 '''
        # 注意力汇聚输出的形状为（批量大小，查询的步数，值的维度）
        # 将特征维度q20与k2映射到同一空间hidden8，再广播
        attention = common.AdditiveAttention( # 初始化加性注意力层
            key_size    =2,     # 键向量维度（与keys的最后一维匹配）
            query_size  =20,    # 查询向量维度（与queries的最后一维匹配）
            num_hiddens =8,     # 隐藏层大小（注意力计算空间维度）
            dropout     =0.1)   # 注意力权重随机丢弃率

        attention.eval() # 设置为评估模式（关闭dropout）
        output = attention(queries, keys, values, valid_lens)
        print(f"加性注意力输出结果:\n{output}")
        common.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                          xlabel='Keys', ylabel='Queries')
    # learn_AdditiveAttention()

    def learn_DotProductAttention():
        ''' 放缩点积注意力机制 '''
        # 查询从均值为0，标准差为1的正态分布中随机抽取指来初始化
        # 批次为2即两个样本，每个样本1个查询即1个序列，序列特征即查询的特征维度为2(与键的特征维度相同)
        queries = torch.normal(0, 1, (2, 1, 2))
        # 训练时会随机丢弃50%的注意力权重（当前eval模式关闭）
        attention = common.DotProductAttention(dropout=0.5) # 初始化放缩点积注意力层
        attention.eval() # 设置为评估模式（关闭dropout等训练专用操作）
        output = attention(queries, keys, values, valid_lens) # 前向传播计算
        print(f"放缩点积注意力输出结果:\n{output}")

        common.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                          xlabel='Keys', ylabel='Queries')
    # learn_DotProductAttention()
# attention_scoring_function()


def learn_Bahdanau_attention():
    ''' Bahdanau 注意力 '''
    # 创建编码器
    # 将长度可变序列 → 固定形状的编码状态
    encoder = common.Seq2SeqEncoder(vocab_size=10,  # 词表大小 即 输入维度为10
                                    embed_size=8,   # 每个单词被表示为8维的向量
                                    num_hiddens=16, # 隐藏层的维度 即 单个隐藏层的神经元数量
                                    num_layers=2)   # 隐藏层的堆叠次数
    encoder.eval() # 设置为评估模式（关闭dropout等训练专用层）

    # 创建带注意力的解码器
    # 固定形状的编码状态 → 将长度可变序列
    decoder = common.Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.eval() # 设置为评估模式（关闭dropout等训练专用层）

    # 创建输入数据 (batch_size=4, 序列长度=7)
    # 总共4个批次数据，每批次数据的长度 即 时间步 为7
    X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)

    # 编码器处理
    # 这里编码器出来output形状:(batch_size,num_steps,num_hiddens)（PyTorch的GRU默认输出格式）
    # state形状:(num_layers,batch_size,num_hiddens)
    enc_outputs = encoder(X)  # outputs: (4,7,16), hidden_state: (2,4,16)

    # 初始化解码器状态
    # 将编码器的输出转换成 解码器所需的状态
    state = decoder.init_state(enc_outputs, None) # outputs调整为(7,4,16)
    output, state = decoder(X, state) # 前向传播：解码（假设输入X作为初始输入）

    # 检查输出维度和状态结构
    print(f"投影到词表空间的所有时间步输出形状：{output.shape}") # 预期: torch.Size([4, 7, 10])
    print(f"更新后的解码器状态三元组长度：{len(state)}") # 3: [enc_outputs, hidden_state, enc_valid_lens]
    print(f"编码器输出: {state[0].shape}")       # torch.Size([4, 7, 16])
    print(f"解码器隐藏状态的层数：{len(state[1])}")
    print(f"首层隐藏状态形状: {state[1][0].shape}")  # torch.Size([4, 16])


    # 词嵌入维度，隐藏层维度，rnn层数，失活率
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10 # 批量大小，序列最大长度
    lr, num_epochs, device = 0.005, 250, common.try_gpu() # 学习率，训练轮数，设备选择

    """ 数据预处理
    train_iter: 训练数据迭代器（自动进行分词、构建词汇表、填充和批处理）
    src_vocab/tgt_vocab: 源语言和目标语言的词汇表对象（包含词元到索引的映射）
    """
    train_iter, src_vocab, tgt_vocab = common.load_data_nmt(downloader, batch_size, num_steps)

    # 模型构建
    """
    编码器结构：
    - 嵌入层：将词元索引映射为32维向量
    - GRU层：2层堆叠，每层32个隐藏单元，带0.1的dropout
    - 输出：所有时间步的隐藏状态（用于注意力计算）和最终隐藏状态（初始化解码器）
    """
    encoder = common.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    """
    解码器核心创新点：
    - 注意力机制：加性注意力（Bahdanau风格），在每个时间步动态生成上下文向量
    - 输入拼接：词嵌入（32维）与上下文向量（32维）拼接为64维输入
    - GRU层：与编码器维度对齐，保持2层堆叠结构
    """
    decoder = common.Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = common.EncoderDecoder(encoder, decoder)  # 封装编码器-解码器结构

    """ 模型训练
    训练过程特点：
    - 损失函数：交叉熵损失（忽略填充符）
    - 优化器：Adam
    - 正则化：梯度裁剪（防梯度爆炸）+ Dropout
    - 教师强制（Teacher Forcing）：训练时使用真实标签作为输入
    """
    common.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 推理与评估
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        # 带注意力可视化的预测
        translation, dec_attention_weight_seq = common.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        # BLEU-2评估（双词组匹配精度）
        print(f'{eng} => {translation}, ',
              f'bleu {common.bleu(translation, fra, k=2):.3f}')

    # 注意力权重可视化处理
    """
    数据处理逻辑：
    遍历dec_attention_weight_seq（解码器各时间步的注意力权重序列）
    step[0][0][0]即首个batch、首个头、首个查询位置对应的所有键的权重（形状为(key_len,)）
    1. 提取每个时间步的注意力权重矩阵（batch_size=1, num_heads=1, query_pos, key_pos）
    2. .cat(, 0)沿时间维度拼接所有权重矩阵
    3. .reshape()调整形状为(1, 1, num_queries, num_keys)
    """
    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((
        1, 1, -1, num_steps))

    """ 绘制注意力热图
    可视化说明：
    - 横轴：源语言句子位置（编码器时间步）
    - 纵轴：目标语言生成位置（解码器时间步）
    - 颜色深浅：注意力权重大小（红色越深表示关注度越高）
    - 典型对齐模式：对角线（单调对齐）、斜线（跨词对齐）
    """
    # 加上一个包含序列结束词元
    common.show_heatmaps(
        attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
        xlabel='Key positions', ylabel='Query positions')
# learn_Bahdanau_attention()


def learn_MultiHeadAttention():
    ''' 学习 多头注意力机制 '''
    num_hiddens, num_heads = 100, 5 # 隐藏层维度，注意力头数
    attention = common.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
    attention.eval() # 设置为评估模式（关闭dropout等训练专用层）
    print(f"[多头注意力]模型结构概览：\n{attention}")

    batch_size, num_queries = 2, 4 # 批量大小，查询数量
    # 键值对数，序列有效长度
    # (因为批量大小为2，所以第一个批量的序列长度中有效长度为3，第二个批量的序列长度中有效长度为2)
    # 序列长度：对于X来说是查询数4，对于Y来说是键值对数6
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2]) # 第二个查询只关注前2个键值对

    # 输入数据：全1张量用于测试
    # X的形状是(batch_size, num_queries, num_hiddens)
    # 即 每个样本有num_queries个查询向量，每个向量维度是num_hiddens
    # Y的形状是(batch_size, num_kvpairs, num_hiddens)
    # 即 每个样本有num_kvpairs个键值对，每个键和值的维度也是num_hiddens
    X = torch.ones((batch_size, num_queries, num_hiddens)) # 查询向量
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens)) # 键值对

    # 前向传播
    output = attention(X, Y, Y, valid_lens)
    print(f"输出形状：{output.shape}")
# learn_MultiHeadAttention()


def Self_attention_and_position_encoding():
    ''' 自注意力和位置编码 '''
    # 自注意力（qkv皆为同一序列）
    num_hiddens, num_heads = 100, 5 # 隐藏层维度，注意力头数
    attention = common.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                       num_hiddens, num_heads, 0.5)
    attention.eval() # 设置为评估模式（关闭dropout等训练专用层）
    print(f"模型结构：\n{attention}")

    # 批量大小，查询数量，序列有效长度(对于X来说，序列长度是键值对数)
    batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
    # 输入数据：全1张量用于测试
    # 张量形状（批量大小，时间步的数目或词元序列的长度，d）
    # 查询数量，等同于序列长度（时间步数/词元数），表示每个序列包含的词元数量（此处为4）
    # d：隐藏层维度（即特征向量长度）
    X = torch.ones((batch_size, num_queries, num_hiddens))
    output = attention(X, X, X, valid_lens)
    print(f"输出形状：{output.shape}")


    encoding_dim, num_steps = 32, 60 # 嵌入维度，序列步长
    pos_encoding = common.PositionalEncoding(encoding_dim, 0) # 创建位置编码器
    pos_encoding.eval() # 设置为评估模式（关闭dropout等训练专用层）

    # 生成全零输入（模拟嵌入向量）
    # ①torch.zeros((1, num_steps, encoding_dim)) 和
    # ②torch.zeros(1, num_steps, encoding_dim) 完全等效，无功能差异
    # ①是 元组形式   ，符合函数参数传递的通用规范
    # ②是 直接参数形式，直接传递多个整数参数，PyTorch内部会自动将其转换为形状元组
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    # 外部再次提取操作：服务于后续处理需求
    P = pos_encoding.P[:, :X.shape[1], :] # 提取实际使用的位置编码

    ''' 可视化特定维度的位置编码变化
    P[0, :, 6:10]取第一个批次(batch维度索引0)，该批次中的所有词元(序列维度全部保留，从0到seq_len-1)
          取每个词元的特征维度中索引6到9的4个维度（对应隐藏层维度索引6,7,8,9）
          
    加转置的原因：调整数据的维度，使能正确将每个特征维度作为单独的曲线绘制，而非将每个位置作为曲线
        若原始数据是：
            位置0: [v6, v7, v8, v9]
            位置1: [v6, v7, v8, v9]
        转置后变成：
            特征6: [位置0的值, 位置1的值, ...]
            特征7: [位置0的值, 位置1的值, ...]
    这样，每条曲线就是某个特征维度随位置的变化，符合常见的可视化需求
    
    legend=["Col %d" % d for d in torch.arange(6, 10)] 等价于
    legend=[f"Col {d}" for d in range(6, 10)]
    '''
    common.plot(torch.arange(num_steps),    # x轴：0~59的位置索引
                P[0, :, 6:10].T,            # 选取维度6~9的4个特征
                xlabel='Row (position)',    # x轴标签
                figsize=(6, 2.5),           # 图像尺寸
                legend=["Col %d" % d for d in torch.arange(6, 10)]) # 图例

    # 打印0到7的二进制表示形式，演示观点：随着编码维度增加，比特值的交替频率正在单调降低
    for i in range(8):
        print(f'{i}的二进制是：{i:>03b}') # 格式化输出二进制（3位宽度右对齐）

    # 热图可视化：展示位置编码矩阵的全局模式
    # 取出第一个批次(batch维度索引0)的所有词元的和所有特征维度，在第0维的位置插入两个长度为1的维度
    # 插入的两个长度为1的维度是指子图行数和列数，
    # 多个子图指的是将多个不同的热图子图显示在同一个画框（即一个图形窗口）内，而这里只有一个子图热图
    P = P[0, :, :].unsqueeze(0).unsqueeze(0) # 调整维度为(1,1,seq_len,dim)
    common.show_heatmaps(P,
                         xlabel='Column (encoding dimension)', # x轴：编码维度
                         ylabel='Row (position)',              # y轴：序列位置
                         figsize=(3.5, 4),
                         cmap='Blues') # 蓝色系配色
# Self_attention_and_position_encoding()


def test_PositionWise_FFN():
    ''' 测试：基于位置的前馈网络
    (对每个输入的x都独立应用相同的线性+激活+线性) '''
    # 创建FFN实例：输入维度4，隐藏层8，输出维度8（实际常用隐藏层维度远大于输入/输出）
    # 注：此处隐藏层维度设为4仅为示例，实际常设为2048等大值
    ffn = common.PositionWiseFFN(4, 4, 8)
    ffn.eval() # 设置为评估模式（关闭dropout等训练专用层）

    # 创建输入数据：2个样本，每个样本3个位置，每个位置4维特征
    input_tensor = torch.ones((2, 3, 4))  # 全1张量用于测试
    output = ffn(input_tensor)      # 执行前向传播
    first_sample_output = output[0] # 获取第一个样本的所有位置输出（形状：3个位置 × 8维输出）
    print("输入形状:", input_tensor.shape)  # torch.Size([2, 3, 4])
    print("输出形状:", output.shape)        # torch.Size([2, 3, 8])
    print("首样本输出:\n", first_sample_output)
# test_PositionWise_FFN()

def test_Add_and_Norm():
    ''' 测试：残差连接和层归一化 模块
    残差输入x经过随机失活后，与子层输出y相加，
    相加后再经过层归一化(样本内所有特征经过归一化，即每个学生的所有科目成绩归一化)'''
    ln = nn.LayerNorm(2)    # 创建【层】归一化对象（对每个样本的所有特征归一化）
    bn = nn.BatchNorm1d(2)  # 创建【批】归一化对象（对每个特征跨样本归一化）

    # 创建输入数据：2个样本，每个样本2个特征
    X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    print("在训练模式下计算X的均值和方差:\n")
    # 层归一化计算（每个样本独立归一化）
    print('layer norm:', ln(X))  # 每个样本的均值和方差独立计算
    # 批归一化计算（跨样本归一化）
    print('batch norm:', bn(X))  # 所有样本的同一特征共享均值和方差

    # AddNorm模块使用示例
    # 创建AddNorm实例：输入张量最后维度为4，Dropout概率0.5
    add_norm = common.AddNorm([3, 4], 0.5)
    add_norm.eval() # 切换至评估模式（关闭Dropout，BatchNorm使用移动平均）

    # 创建输入数据：2个样本，每个样本3个位置，每个位置4维特征
    input1 = torch.ones((2, 3, 4))
    input2 = torch.ones((2, 3, 4))  # 残差连接输入
    output = add_norm(input1, input2) #前向传播
    print("输出形状验证（应与输入相同）\n"
          "输出形状:", output.shape)  # torch.Size([2, 3, 4])
# test_Add_and_Norm()



def test_transformer_encoder():
    ''' 测试：transformer的编码器块及编码器 '''
    # 测试代码（验证维度变换）
    X = torch.ones((2, 100, 24)) # 模拟输入 [batch_size=2, seq_length=100, dim=24]
    valid_lens = torch.tensor([3, 2]) # 有效长度掩码

    # 创建编码器块
    encoder_blk = common.EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval() # 设置为评估模式（关闭dropout等训练专用层）
    output = encoder_blk(X, valid_lens)
    print(f"编码器块输出形状：{output.shape}")

    # 创建完整编码器
    encoder = common.TransformerEncoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval() # 设置为评估模式（关闭dropout等训练专用层）
    output = encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)
    print(f"编码器输出形状：{output.shape}")
# test_transformer_encoder()


def test_transformer_encoder():
    ''' 测试：transformer的解码器块 '''
    # 创建编码器块
    encoder_blk = common.EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval() # 设置为评估模式（关闭dropout等训练专用层）

    # 创建解码器块
    decoder_blk = common.DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval() # 设置为评估模式（关闭dropout等训练专用层）

    X = torch.ones((2, 100, 24)) # 模拟输入 [batch_size=2, seq_length=100, dim=24]
    valid_lens = torch.tensor([3, 2]) # 有效长度掩码

    state = [encoder_blk(X, valid_lens), valid_lens, [None]] # 构建解码器状态
    output = decoder_blk(X, state) # 前向传播
    print(f"解码器块输出，第一个批次的形状：{output[0].shape}") # torch.Size([2, 100, 24])
# test_transformer_encoder()


# 配置超参数
num_hiddens = 32  # 隐藏层维度（Transformer特征维度）
num_layers = 2  # 编码器/解码器堆叠层数
dropout = 0.1  # 随机失活概率（防止过拟合）
batch_size = 64  # 训练批次大小
num_steps = 10  # 序列最大长度（防止过长序列）
lr = 0.005  # 学习率（Adam优化器）
num_epochs = 200  # 训练轮次
device = common.try_gpu()  # 自动选择GPU/CPU

# 前馈网络参数
ffn_num_input = 32  # 前馈网络输入维度（等于num_hiddens）
ffn_num_hiddens = 64  # 前馈网络中间层维度（通常为4倍输入维度）
num_heads = 4  # 多头注意力头数

# 注意力机制参数
key_size = query_size = value_size = 32  # K/Q/V向量维度
norm_shape = [32]  # 层归一化维度（与num_hiddens一致）

# 加载机器翻译数据集（中英翻译示例）
# 返回：数据迭代器、源语言词汇表、目标语言词汇表
train_iter, src_vocab, tgt_vocab = common.load_data_nmt(downloader, batch_size, num_steps)

# 构建Transformer编码器
encoder = common.TransformerEncoder(
    vocab_size=len(src_vocab),    # 源语言词汇表大小
    key_size=key_size,             # 键向量维度
    query_size=query_size,         # 查询向量维度
    value_size=value_size,         # 值向量维度
    num_hiddens=num_hiddens,       # 隐藏层维度
    norm_shape=norm_shape,         # 层归一化形状
    ffn_num_input=ffn_num_input,   # 前馈网络输入维度
    ffn_num_hiddens=ffn_num_hiddens, # 前馈网络中间层维度
    num_heads=num_heads,           # 多头注意力头数
    num_layers=num_layers,         # 编码器层数
    dropout=dropout               # 随机失活概率
)

# 构建Transformer解码器
decoder = common.TransformerDecoder(
    vocab_size=len(tgt_vocab),    # 目标语言词汇表大小
    # 其他参数与编码器配置一致
    key_size=key_size,
    query_size=query_size,
    value_size=value_size,
    num_hiddens=num_hiddens,
    norm_shape=norm_shape,
    ffn_num_input=ffn_num_input,
    ffn_num_hiddens=ffn_num_hiddens,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout
)

# 组合编码器-解码器架构
net = common.EncoderDecoder(encoder, decoder)

# 训练序列到序列模型
# 模型，训练数据迭代器，学习率，训练轮次，目标语言词汇表（用于评估），训练设备
common.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# 英法对照测试集
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras): # 逐句进行机器翻译并评估
    # 预测翻译结果（包含注意力权重序列）
    translation, dec_attention_weight_seq = common.predict_seq2seq(
        net,           # 训练好的seq2seq模型
        eng,           # 待翻译的英文句子
        src_vocab,     # 源语言词汇表
        tgt_vocab,     # 目标语言词汇表
        num_steps,     # 最大序列长度
        device,        # 计算设备（GPU/CPU）
        True           # 返回注意力权重
    )
    # 计算BLEU-2分数（双语评估替手）
    print(f'{eng} => {translation}, ',
          f'bleu {common.bleu(translation, fra, k=2):.3f}')

# 提取编码器各层注意力权重
enc_attention_weights = torch.cat(net.encoder.attention_weights, 0)  # 拼接所有层
enc_attention_weights = enc_attention_weights.reshape((
    num_layers,    # 2层编码器
    num_heads,     # 4个注意力头
    -1,            # 自动计算维度（源序列长度）
    num_steps      # 目标序列长度
))
print(f"编码器注意力权重：{enc_attention_weights.shape}")

# 可视化编码器自注意力热图
common.show_heatmaps(
    enc_attention_weights.cpu(),  # 转为CPU张量
    xlabel='Key positions',  # 横轴：键位置
    ylabel='Query positions',  # 纵轴：查询位置
    titles=['Head %d' % i for i in range(1, 5)],  # 4个头的标题
    figsize=(7, 3.5)  # 图像尺寸
)

# 解码器注意力权重处理
# 二维列表构建：按时间步、层、注意力类型、头展开权重
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq # 遍历时间步
                            for attn in step    # 遍历解码器层（attn，即每层的注意力数据）
                            for blk in attn     # 遍历注意力类型（blk，即 自注意力/交叉注意力）
                            for head in blk]    # 遍历注意力头

# 转换为DataFrame并填充缺失值
# 缺失值处理：使用fillna(0.0)将NaN填充为0
dec_attention_weights_filled = torch.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)

# 重塑为5维张量：[时间步, 2种类型, 层数, 头数, 序列长度]
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))

# 分离两种注意力类型
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4) # 调整维度顺序
# 打印维度确认
print(f"自注意力：\n{dec_self_attention_weights.shape}")
print(f"交叉注意力：\n{dec_inter_attention_weights.shape}")

# Plusonetoincludethebeginning-of-sequencetoken
# 可视化解码器自注意力热图（包含起始符）
common.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

# 可视化解码器交叉注意力热图（编码器-解码器）
common.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))


plt.pause(4444)  # 间隔的秒数： 4s




