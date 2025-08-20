import torch
from torch import nn
import common
from torch.nn import functional as F


# 下载器与数据集配置
# 为 time_machine 数据集注册下载信息，包括文件路径和校验哈希值（用于验证文件完整性）
downloader = common.C_Downloader()
DATA_HUB = downloader.DATA_HUB  # 字典，存储数据集名称与下载信息
DATA_URL = downloader.DATA_URL  # 基础URL，指向数据集的存储位置

DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


batch_size, num_steps = 32, 35  # 每个小批量包含32个子序列，每个子序列的词元数为35
train_iter, vocab = common.load_data_time_machine(downloader, batch_size, num_steps)  # 词表对象


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
    X: 输入序列，形状为 (批量大小, 时间步数量)，每个元素是词索引（整数）
    state: 隐藏状态
    返回:输出和新的隐藏状态
    __call__实现后，该类的实例就可以被当作函数使用，
    即通过`instance(arguments)`的方式调用触发`__call__`方法的执行
    这里：创建RNNModelScratch的实例（例如net）后，可以像函数一样调用这个实例👇
    net = RNNModelScratch(len(vocab), num_hiddens, common.try_gpu(), get_params,
                        init_rnn_state, rnn) # 创建模型实例
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


# 门控循环单元（GRU）的从零开始实现
def learn_gru_StartFromScratch():
    # 1、初始化模型参数
    """
    初始化 门控循环单元GRU 的模型参数
    权重使用小随机数初始化，偏置初始化为0
    包括：输入到隐藏层、隐藏层到隐藏层、隐藏层到输出层的权重以及相应的偏置
    参数:
    vocab_size : 词表大小 (输入和输出的维度)
    num_hiddens: 隐藏层大小
    device     : 计算设备 (CPU/GPU)
    返回:
    params: 包含所有参数的列表
    """
    def get_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size # 输入和输出的维度都是词表大小（字符级）

        def normal(shape):  # 定义正态分布初始化函数
            """生成服从正态分布的随机张量"""
            # 生成服从正态分布的随机参数，乘以0.01使其值较小，有利于训练稳定性
            return torch.randn(size=shape, device=device)*0.01 # 生成小随机数

        def three(): # 生成与门控和候选隐状态相关的三组参数（输入权重、循环权重、偏置）
            """返回三个参数组：(输入权重, 循环权重, 偏置)"""
            return (normal((num_inputs, num_hiddens)),       # 输入到隐藏层的权重
                    normal((num_hiddens, num_hiddens)),      # 隐藏层到隐藏层的权重
                    torch.zeros(num_hiddens, device=device)) # 偏置（初始化为0）

        # 为GRU的三个核心组件（更新门、重置门、候选状态）分别创建参数
        W_xz, W_hz, b_z = three()  # 更新门参数 (Z门) : 控制新旧状态混合比例
        W_xr, W_hr, b_r = three()  # 重置门参数 (R门) : 控制历史信息重置程度
        W_xh, W_hh, b_h = three()  # 候选隐状态参数 : 计算临时新状态

        # 输出层参数单独初始化 : 将隐状态映射到输出空间
        W_hq = normal((num_hiddens, num_outputs))       # 隐藏层到输出层的权重
        b_q = torch.zeros(num_outputs, device=device)   # 输出层偏置
        # 将所有参数放入列表，并附加梯度（组合所有参数并启用梯度追踪）
        params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
        for param in params: # 设置为需要梯度
            param.requires_grad_(True)  # 启用自动微分(开启梯度追踪)
        return params # 返回模型参数列表

    # 2、定义训练
    '''
    初始化GRU的隐藏状态
    功能：返回初始的隐藏状态（一个全零张量），形状为 (batch_size, num_hiddens)
    batch_size : 批量大小
    num_hiddens: 隐藏层大小
    device     : 计算设备
    返回:包含初始隐藏状态的元组（全零张量）(H0,)
    说明：在训练开始时或处理新序列时，隐状态需要初始化为零
    '''
    def init_gru_state(batch_size, num_hiddens, device):
        # 初始化隐状态为全0，形状为 (batch_size, num_hiddens隐藏单元数)
        return (torch.zeros((batch_size, num_hiddens), device=device), ) # 初始隐藏状态为全0

    '''
    GRU前向传播计算
    每个时间步的输出是线性变换后的隐状态（未经过softmax，因为训练时使用交叉熵损失函数会包含softmax）
    inputs: 输入序列 (时间步数量, 批量大小, 词表大小)
    state : 初始隐藏状态的元组 (H0,)
    params: 模型参数列表 [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    返回:
    outputs  : 所有时间步的输出，形状为 (时间步数量 * 批量大小, 词表大小)
    (H,): 更新后的最终隐状态
    '''
    def gru(inputs, state, params):
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params # 解包参数
        H, = state       # 解包当前隐藏状态 (batch_size, num_hiddens)
        outputs = []     # 用于存储每个时间步的输出
        for X in inputs: # 遍历输入序列的每个时间步
            # 更新门计算：决定保留多少旧状态 (控制状态更新程度)
            # Z_t = σ(X_t * W_xz + H_{t-1} * W_hz + b_z)
            Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z) # 形状: (batch_size, num_hiddens)

            # 重置门计算：决定重置多少历史信息
            # R_t = σ(X_t * W_xr + H_{t-1} * W_hr + b_r)
            R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r) # 形状: (batch_size, num_hiddens)

            # 候选隐状态计算：（使用重置门控制历史信息影响）
            # \tilde{H}_t = tanh(X_t * W_xh + (R_t ⊙ H_{t-1}) * W_hh + b_h)
            H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h) # 形状: (batch_size, num_hiddens)

            # 更新最终隐状态：混合旧状态和新候选状态：H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ \tilde{H}_t
            H = Z * H + (1 - Z) * H_tilda # 形状: (batch_size, num_hiddens)

            # 计算当前时间步输出，即 输出层：Y_t = H_t * W_hq + b_q
            Y = H @ W_hq + b_q            # 形状: (batch_size, vocab_size)
            outputs.append(Y)  # 保存输出
        # 沿时间步维度拼接所有输出（形状：时间步数×批量大小×词表大小）
        # (H,) 表示只包含一个元素H的元组（tuple），但真正定义元组的是逗号，而不是圆括号
        # 如果写(H)，这只是带括号的表达式，等同于`H`，而不是元组。为了表示这是只有一个元素的元组，需在元素后加逗号。所以：
        # (H)  是一个表达式，其值为H
        # (H,) 是一个元组，包含一个元素H
        return torch.cat(outputs, dim=0), (H,) # 返回输出和最后一个隐藏状态

    # 3、训练与预测
    # 词表大小、隐藏层大小256、设备
    vocab_size, num_hiddens, device = len(vocab), 256, common.try_gpu()
    num_epochs, lr = 500, 1 # 训练周期即迭代周期为500，即训练500轮；学习率为1（较高学习率因从0实现）
    model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                                init_gru_state, gru) # 创建模型实例
    common.train_ch8(model, train_iter, vocab, lr, num_epochs, device) # 训练模型
# learn_gru_StartFromScratch()



# 定义完整的RNN模型类
# rnn_layer只包含隐藏的循环层，因此另外还需要创建一个单独的输出层
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer                    # 传入的RNN层
        self.vocab_size = vocab_size            # 词表大小
        self.num_hiddens = self.rnn.hidden_size # 从RNN层获取隐藏单元数

        # 判断RNN是否为双向：若RNN是双向(之后将介绍)，num_directions为2，否则为1
        if not self.rnn.bidirectional: # 单向RNN: num_directions = 1
            self.num_directions = 1
            # 线性层: 将隐藏状态映射到词表大小
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:                          # 双向RNN: num_directions = 2
            self.num_directions = 2
            # 对于双向RNN，需要将两个方向的隐藏状态拼接
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    """ 前向传播函数
    inputs: 输入张量，形状为(批量大小, 时间步数)
    state: 隐藏状态
    """
    def forward(self, inputs, state):
        # 输入转为one-hot编码
        # inputs.T: 转置为(时间步数, 批量大小)
        # .long() ：将张量转换为长整型（int64），F.one_hot要求索引为整型，而.long()确保这一点
        # one-hot编码后形状: (时间步数, 批量大小, 词表大小)
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)         # 独热张量转换为float32 (PyTorch的线性层需要浮点输入)

        # Y: 所有时间步的输出，形状为(时间步数, 批量大小, 隐藏单元数 * 方向数)
        # state: 更新后的隐藏状态
        Y, state = self.rnn(X, state)   # 通过RNN层

        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 即 重塑Y的形状为 (时间步数 * 批量大小, 隐藏单元数 * 方向数)
        # 这样每个时间步的每个样本都可以独立处理
        # 其输出形状为 (时间步数*批量大小, 词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1]))) # 通过全连接层进行预测
        return output, state

    # 初始化隐藏状态函数：根据RNN类型(GRU/LSTM)返回适当形式的初始状态
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM): # 对于GRU类型的RNN
            # nn.GRU以张量作为隐状态
            # 返回零张量作为初始状态，形状: (层数 * 方向数, 批量大小, 隐藏单元数)
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else: # 对于LSTM类型的RNN
            # nn.LSTM以元组作为隐状态，LSTM需要两个状态: (隐藏状态h, 细胞状态c)
            return (
                torch.zeros(( # 隐藏状态
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                torch.zeros(( # 细胞状态
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


# 门控循环单元（GRU）的简洁实现
def learn_gru_SimpleImplementation():
    # 词表大小、隐藏层大小256、设备
    vocab_size, num_hiddens, device = len(vocab), 256, common.try_gpu()
    num_epochs, lr = 500, 1 # 训练周期即迭代周期为500，即训练500轮；学习率为1(使用内置GRU时通常需要调整，这里保持与从零实现一致)

    num_inputs = vocab_size # 输入维度等于词表大小（字符级one-hot表示）
    """
    nn.GRU关键参数:
    - num_inputs: 输入特征维度 (词表大小)
    - num_hiddens: 隐藏层神经元数量 (256)
    - 默认: 单层、非双向、batch_first=False (序列维度在前)
    """
    gru_layer = nn.GRU(num_inputs, num_hiddens) # 创建内置GRU层：输入维度, 隐藏层维度
    """
    RNNModel类功能:
    1. 处理输入数据的one-hot编码
    2. 通过GRU层计算隐藏状态
    3. 添加全连接输出层 (隐藏层->词表)
    4. 管理隐藏状态的初始化和传递
    """
    model = RNNModel(gru_layer, len(vocab)) # 传入GRU层和词表大小
    model = model.to(device) # 将模型移到指定设备 (GPU/CPU)
    common.train_ch8(model, train_iter, vocab, lr, num_epochs, device) # 训练模型
# learn_gru_SimpleImplementation()



# 长短期记忆网络（LSTM）的从零开始实现
def learn_LSTM_StartFromScratch():
    # 1、初始化模型参数
    """
    初始化 长短期记忆网络LSTM 所有可训练参数
    权重使用小随机数初始化，偏置初始化为0
    参数:
    vocab_size : 词表大小 (输入和输出的维度)
    num_hiddens: 隐藏层数量
    device     : 计算设备 (CPU/GPU)
    返回:
    params: 包含所有参数的列表
    """
    def get_lstm_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size # 输入和输出的维度都是词表大小（字符级语言模型）

        def normal(shape): # 定义正态分布初始化函数
            """生成服从正态分布的随机张量（缩小初始值范围）"""
            # 生成服从正态分布的随机参数，乘以0.01使其值较小，有利于训练稳定性
            return torch.randn(size=shape, device=device)*0.01 # 生成小随机数

        def three():
            """返回三个参数组：(输入权重, 循环权重, 偏置)"""
            return (normal((num_inputs, num_hiddens)),        # 输入到隐藏层的权重
                    normal((num_hiddens, num_hiddens)),       # 隐藏层到隐藏层的权重
                    torch.zeros(num_hiddens, device=device))  # 偏置（初始化为0）

        W_xi, W_hi, b_i = three()  # 输入门参数 (控制新信息流入)
        W_xf, W_hf, b_f = three()  # 遗忘门参数 (控制旧信息保留)
        W_xo, W_ho, b_o = three()  # 输出门参数 (控制信息输出)
        W_xc, W_hc, b_c = three()  # 候选记忆元参数 (新记忆计算)
        # 输出层参数单独初始化 : 将隐状态映射到输出空间（词表大小）
        W_hq = normal((num_hiddens, num_outputs))      # 隐藏层到输出层的权重
        b_q = torch.zeros(num_outputs, device=device)  # 输出层偏置（初始化为0）
        # 将所有参数放入列表，并附加梯度（组合所有参数并启用梯度追踪）
        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                  b_c, W_hq, b_q]
        for param in params:  # 设置为需要梯度
            param.requires_grad_(True)  # 启用自动微分(开启梯度追踪)
        return params # 返回模型参数列表

    # 2、定义模型
    '''
    初始化 LSTM的 隐藏状态 和 记忆元状态
    batch_size : 批量大小
    num_hiddens: 隐藏层大小
    device     : 计算设备
    返回:包含(初始隐藏状态, 初始记忆元状态)的元组（全零张量）
    '''
    def init_lstm_state(batch_size, num_hiddens, device):
        # LSTM有两个状态：隐藏状态H和记忆元状态C，初始化为全0，形状为 (batch_size, num_hiddens隐藏单元数)
        return (torch.zeros((batch_size, num_hiddens), device=device), # 隐藏状态 H
                torch.zeros((batch_size, num_hiddens), device=device)) # 记忆元状态 C

    '''
    LSTM前向传播计算
    每个时间步的输出是线性变换后的隐状态（未经过softmax，因为训练时使用交叉熵损失函数会包含softmax）
    inputs: 输入序列 (时间步列表, 每个形状为[batch_size, vocab_size]) 即 (时间步数量, 批量大小, 词表大小)
    state : 初始隐藏状态的元组 (H, C)
    params: 模型参数列表
    返回:
    outputs  : 所有时间步的输出，形状为 (时间步数量 * 批量大小, 词表大小)
    (H, C): 更新后的最终状态
    '''
    def lstm(inputs, state, params):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
         W_hq, b_q] = params # 解包参数 (共14个参数)
        (H, C) = state       # 解包初始状态（H: 隐藏状态, C: 记忆元状态）
        outputs = []         # 用于存储每个时间步的输出
        for X in inputs:     # 遍历输入序列的每个时间步
            # 1. 输入门 (I_t)计算：控制新信息流入
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i) # 形状: (batch_size, num_hiddens)

            # 2. 遗忘门 (F_t)计算：控制旧信息保留
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f) # 形状: (batch_size, num_hiddens)

            # 3. 输出门 (O_t)计算：控制信息输出
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o) # 形状: (batch_size, num_hiddens)

            # 4. 候选记忆元 (C_tilda_t)计算：新信息的原始表示
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c) # 形状: (batch_size, num_hiddens)

            # 5. 更新记忆元 (C_t)状态：遗忘旧信息 + 添加新信息 （遗忘门控制旧状态，输入门控制新候选状态）
            C = F * C + I * C_tilda # 形状: (batch_size, num_hiddens)

            # 6. 更新隐藏状态 (H_t)：基于记忆元生成新隐藏状态（输出门控制tanh(记忆元)的输出）
            H = O * torch.tanh(C) # 形状: (batch_size, num_hiddens)

            # 7. 计算当前时间步输出 (Y_t)
            Y = (H @ W_hq) + b_q  # 形状: (batch_size, vocab_size)
            outputs.append(Y)     # 保存输出
        return torch.cat(outputs, dim=0), (H, C) # 沿时间步维度拼接所有输出（形状：时间步数×批量大小×词表大小）

    # 3、训练和预测
    # 词表大小、隐藏层大小256、设备
    vocab_size, num_hiddens, device = len(vocab), 256, common.try_gpu()
    num_epochs, lr = 500, 1 # 训练周期即迭代周期为500，即训练500轮；学习率为1（较高学习率因从零实现）
    # 创建LSTM模型实例（使用RNNModelScratch类封装，传入LSTM的参数初始化、状态初始化和前向传播函数）
    model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                                init_lstm_state, lstm)
    # 训练模型（train_iter是数据迭代器，vocab是词表）
    common.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# learn_LSTM_StartFromScratch()


# 长短期记忆网络（LSTM）的简洁实现
def learn_LSTM_SimpleImplementation():
    # 词表大小（唯一字符数量）、LSTM隐藏层维度（256个神经元）、设备
    vocab_size, num_hiddens, device = len(vocab), 256, common.try_gpu()
    # 训练周期即迭代周期为500，即训练500轮；
    # 学习率为1(使用内置GRU时通常需要调整，这里保持与从零实现一致，实际使用内置LSTM时建议0.01-0.1)
    num_epochs, lr = 500, 1
    num_inputs = vocab_size # 输入维度等于词表大小（字符级one-hot表示）
    """
    nn.LSTM关键参数:
    - input_size: 输入特征维度 = 词表大小
    - hidden_size: 隐藏状态维度 = 256
    - 默认配置:
      - 单层单向LSTM
      - batch_first=False (输入形状为[seq_len, batch, input_size])
      - 使用tanh激活函数
      - 偏置项默认启用
    """
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = RNNModel(lstm_layer, len(vocab)) # 传入GRU层和词表大小
    """
    RNNModel类封装了:
    1. 输入处理: 将字符索引转换为向量（可能使用嵌入层或one-hot）
    2. LSTM层: 核心序列建模
    3. 输出层: 全连接层 (256隐藏单元 → 词表大小)
    4. 状态管理: 自动处理LSTM的隐藏状态和记忆元状态
    
    预期结构:
    class RNNModel(nn.Module):
        def __init__(self, rnn_layer, vocab_size):
            super().__init__()
            self.rnn = rnn_layer
            self.vocab_size = vocab_size
            self.num_hiddens = rnn_layer.hidden_size
            # 输出层
            self.dense = nn.Linear(self.num_hiddens, vocab_size)
    
        def forward(self, inputs, state):
            # 输入转换 (batch, seq_len) → (seq_len, batch, vocab_size)
            X = F.one_hot(inputs.T.long(), self.vocab_size).float()
            # LSTM计算
            Y, state = self.rnn(X, state)
            # 输出层
            output = self.dense(Y.reshape(-1, Y.shape[-1]))
            return output, state
    
        def begin_state(self, batch_size=1):
            return (torch.zeros(1, batch_size, self.num_hiddens),
                    torch.zeros(1, batch_size, self.num_hiddens))
    """

    model = model.to(device) # 将模型移到指定设备 (GPU/CPU)
    common.train_ch8(model, train_iter, vocab, lr, num_epochs, device) # 训练模型
# learn_LSTM_SimpleImplementation()


# 深度循环神经网络（仅示范简洁实现）
def learn_drnn():
    # 词表大小，隐藏层数量
    # num_layers : LSTM层数 = 2 (堆叠两层LSTM)
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size     # 输入特征维度为词表大小（每个字符用one-hot表示）
    device = common.try_gpu()   # 设备
    # input_size : 输入特征维度 = 词表大小
    # hidden_size: 隐藏状态维度 = 256
    # num_layers : LSTM层数 = 2 (堆叠两层LSTM)
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers) # 创建LSTM层
    model = RNNModel(lstm_layer, len(vocab)) # 创建完整的RNN模型
    model = model.to(device) # 将模型移至指定设备

    num_epochs, lr = 500, 2  # 迭代轮数为500，学习率
    common.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
# learn_drnn()


# 双向循环神经网络 的错误示范
def bidirectionalRNN_incorrect_demonstration():
    # 词表大小，隐藏层数量，LSTM层数为2 (堆叠两层LSTM)
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size     # 输入特征维度为词表大小（每个字符用one-hot表示）
    device = common.try_gpu()   # 设备
    # 通过设置“bidirective=True”启用双向 来定义双向LSTM模型
    # 使用双向LSTM来构建语言模型
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
    '''
    nn.LSTM参数说明:
    - input_size: 输入特征维度 = vocab_size
    - hidden_size: 隐藏状态维度 = 256
    - num_layers: LSTM层数 = 2 (堆叠两层)
    - bidirectional=True: 启用双向处理
    
    双向LSTM的特点:
    1. 每个时间步包含两个隐藏状态:
       - 前向状态: 处理过去→未来的信息
       - 后向状态: 处理未来→过去的信息
    2. 输出维度变化:
       - 单层双向: 隐藏状态维度 = 2 * num_hiddens
       - 多层双向: 输出形状 = (seq_len, batch, 2*num_hiddens)
    '''
    model = RNNModel(lstm_layer, len(vocab)) # 创建完整的RNN模型
    model = model.to(device) # 将模型移至指定设备
    # 训练模型
    num_epochs, lr = 500, 1  # 迭代轮数为500，学习率
    common.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# bidirectionalRNN_incorrect_demonstration()


# 注册数据集信息到DATA_HUB全局字典
# 格式：(数据集URL, MD5校验值)
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip', # 完整下载URL（DATA_URL是d2l定义的基准URL）
                           '94646ad1522d915e7b0f9296181140edcf86a4f5') # 文件MD5，用于校验下载完整性

def learn_MachinaTranslation_and_Data():
    # 载入(神经网络机器翻译nmt中要用到的)“英语－法语”数据集
    raw_text = common.read_data_nmt(downloader) # 此时raw_text包含整个语料库的原始文本
    print(f"数据集预览（显示前75个字符）：\n{raw_text[:75]}")

    text = common.preprocess_nmt(raw_text)
    print(f"预处理后 数据集预览（显示前80个字符）：\n{text[:80]}")

    source, target = common.tokenize_nmt(text)
    print(f"源语言(英语)【前6个】文本序列的词元列表：\n{source[:6]}")
    print(f"目标语言(法语)【前6个】文本序列的词元列表：\n{target[:6]}")

    # 绘制每个文本序列所包含的词元数量的直方图
    # common.show_list_len_pair_hist(['source', 'target'],
    #                                '# tokens per sequence','count',
    #                                source, target)

    # 给 词元化后的源语言 句子列表 构建词表，管理词元与索引的映射关系
    src_vocab = common.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(f"词表大小：{len(src_vocab)}")

    # 打印出元素为整型是因为 src_vocab已将单词映射为整数ID
    print(f"以取第一个句子为例\n{src_vocab[source[0]]}")
    print(f"将文本序列截断或填充至10个后：(取第一个句子为例)\n"
          f"{common.truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])}")

    # 加载“英语－法语”数据集的数据迭代器，以及源语言和目标语言的词表
    train_iter, src_vocab, tgt_vocab = common.load_data_nmt(downloader, batch_size=2, num_steps=8)
    # 读出“英语－法语”数据集中的第一个小批量数据
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('源语言X:', X.type(torch.int32))
        print('源语言X的有效长度:', X_valid_len)
        print('目标语言Y:', Y.type(torch.int32))
        print('目标语言Y的有效长度:', Y_valid_len)
        break
learn_MachinaTranslation_and_Data()




# 用于序列到序列学习（seq2seq）的循环神经网络编码器
class Seq2SeqEncoder(common.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层:获得输入序列中每个词元的特征向量
        # 将词元索引转换为密集向量（vocab_size → embed_size）
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 循环神经网络（GRU）
        # embed_size: 输入特征维度（嵌入层输出维度）
        # num_hiddens: 隐状态维度
        # num_layers: 堆叠的RNN层数
        # dropout: 层间dropout概率（仅在num_layers>1时生效
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args): # 前向传播逻辑
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X) # 嵌入层处理：(bch_sz, num_steps)→ (bch_sz, num_steps, embed_size)

        # permute(1, 0, 2)交换前两维，即 时间步和批次维度
        # 因为在循环神经网络模型中，轴一对应于时间步
        # 即 RNN要求输入形状为 (num_steps, batch_size, embed_size)
        X = X.permute(1, 0, 2) # 形状变为 (num_steps, batch_size, embed_size)

        # RNN处理
        # 默认初始隐状态state=None时，自动初始化为全零
        # 若未提及状态，则默认为0
        output, state = self.rnn(X)

        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state # 所有时间步的隐状态，最后一层的最终隐状态

encoder = Seq2SeqEncoder(vocab_size=10,  # 词表大小 即 输入维度为10
                         embed_size=8,   # 每个单词被表示为8维的向量，即 特征向量的维度
                         num_hiddens=16, # 隐藏层的维度 即 单个隐藏层的神经元数量
                         num_layers=2)   # 隐藏层的堆叠次数为2
encoder.eval() # 设置为评估模式（关闭dropout等训练专用层）
# 模拟输入：4个序列（batch_size=4），每个序列7个时间步（num_steps=7）
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X) # 前向传播
print(f"最后一层的隐状态的输出(所有时间步)：{output.shape}")
print(f"最后一个时间步的多层隐状态的形状：{state.shape}")


# 用于序列到序列学习（seq2seq）的循环神经网络解码器
class Seq2SeqDecoder(common.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    """ 初始化解码器
    vocab_size : 目标语言词表大小
    embed_size : 词嵌入维度
    num_hiddens: GRU隐藏层大小，即 隐藏层维度
    num_layers : GRU层数
    dropout    : 随机失活率（防止过拟合）
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 词嵌入层：将词汇表索引转换为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU解码器核心：
        # 输入维度 = embed_size（当前输入） + num_hiddens（上下文变量）
        # 上下文向量：来自编码器的最终隐藏状态(这种设计允许解码器访问源序列信息)
        # 隐藏层维度 = num_hiddens
        # 层数 = num_layers
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # 输出层：将RNN隐藏状态映射到词汇表空间
        self.dense = nn.Linear(num_hiddens, vocab_size)

    # enc_outputs: 编码器的输出元组 (output, state)
    def init_state(self, enc_outputs, *args):
        """初始化解码器状态（使用编码器最后一层的隐藏状态）"""
        # enc_outputs[1] 是编码器RNN的最终隐藏状态
        # 形状：(num_layers, batch_size, num_hiddens)
        return enc_outputs[1] # 返回:解码器的初始状态（直接使用编码器的最终隐藏状态）

    """
    前向传播逻辑：
    X: 当前时间步的输入（解码器输入序列），形状 (batch_size, num_steps)
    state: 编码器提供的初始隐藏状态（或上一步的隐藏状态）
    返回:
    output: 预测输出，形状 (batch_size, num_steps, vocab_size)
    state: 更新后的隐藏状态
    """
    def forward(self, X, state):
        # 1. 词嵌入：将输入索引转换为向量
        # self.embedding(X) 将输入索引转换为向量，
        #       形状(batch_size,num_steps)→(batch_size,num_steps,embed_size)
        # .permute(1, 0, 2) 将前两个维度位置交换，(PyTorch的RNN要求序列维度在前)
        #       形状(batch_size,num_steps,embed_size)→(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2) # 形状(num_steps,batch_size,embed_size)

        # 2. 准备上下文变量（来自编码器的最终隐藏状态）
        # state[-1] 取最后一层的隐藏状态，形状：(batch_size, num_hiddens)
        # .repeat(X.shape[0], 1, 1)扩展时间步维度，使其与X的num_steps一致
        # 即 沿第一维重复直至扩展为 与X的第0维num_steps同大小
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1) # 形状：(num_steps,batch_size,num_hiddens)

        # 3. 拼接当前输入和上下文变量（沿第3维拼接）
        # X_and_context形状：(num_steps, batch_size, embed_size + num_hiddens)
        X_and_context = torch.cat((X, context), 2)

        # 4. RNN处理：输入拼接后的张量，更新隐藏状态
        # output形状：(num_steps, batch_size, num_hiddens)
        # state形状：(num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X_and_context, state)

        # 5. 生成预测：将RNN输出映射到词汇表空间
        # 调整维度顺序为 (batch_size, num_steps, vocab_size)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)  预测输出
        # state的形状:(num_layers,batch_size,num_hiddens) 更新后的隐藏状态
        return output, state

# 目标词汇大小为10，嵌入维度为8
# 隐藏层维度为16(与编码器隐藏层大小相同)，隐藏层堆叠次数为2(与编码器层数相同)
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval() # 设置为评估模式（关闭dropout等训练专用层）
state = decoder.init_state(encoder(X)) # 初始化解码器状态
output, state = decoder(X, state) # 前向传播
output.shape, state.shape
print(f"解码器的输出(所有时间步)：{output.shape}")
print(f"最后一个时间步的多层隐状态的形状：{state.shape}")


X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"假设两个序列的有效长度（不包括填充词元）分别为1和2 ↓"
      f"\n屏蔽前的原始序列：\n{X}"
      f"\n在序列中屏蔽不相关的项后 的序列效果："
      f"\n{common.sequence_mask(X, torch.tensor([1, 2]))}")

X = torch.ones(2, 3, 4)
print(f"屏蔽最后几个轴上的所有项，且使用指定的非零值-1来替换这些项 ↓"
      f"\n屏蔽前的原始序列：\n{X}"
      f"\n在序列中屏蔽不相关的项后 的序列效果："
      f"\n{common.sequence_mask(X, torch.tensor([1, 2]), value=-1)}")
""" 输出效果解析：
第一个序列 (有效长度1):
  [
    [1,1,1,1],  # 保留
    [-1,-1,-1,-1], # 掩码
    [-1,-1,-1,-1]  # 掩码
  ]
第二个序列 (有效长度2):
  [
    [1,1,1,1],  # 保留
    [1,1,1,1],  # 保留
    [-1,-1,-1,-1] # 掩码
  ]
"""

loss = common.MaskedSoftmaxCELoss() # 创建损失函数实例
# 计算损失
result = loss(
     torch.ones(3, 4, 10),  # pred：3个序列，4个时间步，10个词汇表
     torch.ones((3, 4), dtype=torch.long),  # label：所有标签为1
     torch.tensor([4, 2, 0]))               # valid_len：有效长度
print(f"计算损失：{result}")


# 词嵌入维度32，隐藏层维度32，rnn层数2，随机失活率0.1
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10 # 批量大小和序列长度
lr, num_epochs, device = 0.005, 300, common.try_gpu() # 学习率，训练轮数

# 加载数据（法语→英语翻译示例）
# 数据迭代器，源语言的词表，目标语言的词表
train_iter, src_vocab, tgt_vocab = common.load_data_nmt(downloader, batch_size, num_steps)

# 定义编码器（双向GRU）和解码器（单向GRU）
encoder = Seq2SeqEncoder(len(src_vocab), # 源语言词表大小
                        embed_size, num_hiddens, num_layers,dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), # 目标语言词表大小
                        embed_size, num_hiddens, num_layers,dropout)
net = common.EncoderDecoder(encoder, decoder) # 组合为Seq2Seq模型
common.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device) # 启动训练

# 英语句子列表（源语言）
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# 对应的法语句子列表（目标语言/参考译文）
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras): # 遍历每个英语-法语句子对
    # 使用seq2seq模型进行翻译预测
    translation, attention_weight_seq = common.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    # 计算并打印BLEU分数（使用1-gram和2-gram）
    bleu_score = common.bleu(translation, fra, k=2)
    # 输出结果：原句 => 预测翻译, BLEU分数
    print(f'源({eng}) => 预测({translation}), 相似度评估bleu {bleu_score:.3f}')




