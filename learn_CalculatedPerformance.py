import torch
from torch import nn
from torch.nn import functional as F
import common

import os
import subprocess
import numpy

def learn_Compilers_and_interpreters():
    '''章节12.1：编译器和解释器'''
    def test_DemonstrateImperativeProgramming():
        '''演示命令式编程'''
        def add(a, b):
            return a + b

        def fancy_func(a, b, c, d):
            e = add(a, b)
            f = add(c, d)
            g = add(e, f)
            return g

        print(fancy_func(1, 2, 3, 4))
    # test_DemonstrateImperativeProgramming()

    def test_SimulationSymbolicProgramming():
        '''模拟符号式编程
        通过字符串拼接生成Python代码，然后动态编译执行
        演示了符号式编程的构建计算图思想
        执行流程：字符串代码 → compile() → 字节码 → exec() → 运行结果
        '''
        def add_():
            '''生成加法函数的字符串代码
            返回一个字符串，内容是一个加法函数的定义
            '''
            return '''
        def add(a, b):
            return a + b
        '''

        def fancy_func_():
            '''生成复杂函数的字符串代码（调用add函数）
            数学运算：fancy_func(a,b,c,d) = (a+b) + (c+d)
            '''
            return '''
        def fancy_func(a, b, c, d):
            e = add(a, b)
            f = add(c, d)
            g = add(e, f)
            return g
        '''

        def evoke_():
            '''组合生成完整程序
            组合所有代码片段并添加执行语句 '''
            return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

        prog = evoke_() # 生成完整的Python程序字符串（定义计算图：先构建计算关系）
        print(prog)     # 打印生成的代码
        y = compile(prog, '', 'exec') # 将字符串代码编译为字节码（图优化：编译阶段）
        exec(y)         # 执行编译后的代码（后执行计算）
    # test_SimulationSymbolicProgramming()

    # 生产网络的工厂模式
    def get_net():
        '''网络工厂函数：封装模型创建逻辑
        工厂模式优势：
            封装性：隐藏模型构建细节
            可复用：多处调用保证一致性
            易维护：修改只需改一个地方
        网络架构：
        输入(512) → 全连接(256) → ReLU → 全连接(128) → ReLU → 输出(2)
        '''
        net = nn.Sequential(nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2)) # 输出层: 128→2 (二分类)
        return net

    x = torch.randn(size=(1, 512))  # 创建测试输入: batch_size=1, 特征数=512
    net = get_net()                 # 工厂函数创建模型
    print(f"原始模型输出: \n{net(x)}")  # 前向传播测试
    # 此时模型状态：动态图模式，Python解释器逐层执行

    net = torch.jit.script(net)  # 将模型编译为TorchScript
    print(f"使用torch.jit.script函数来转换模型：\n{net(x)}")
    '''编译过程：
    内部发生的变化：
    动态Python代码 → TorchScript编译器 → 优化后的静态计算图
    '''

    '''
    # with语句的执行流程：
    with Benchmark('测试名称') as bench:
        # 1. 进入时：调用__enter__()，启动计时器
        # 2. 执行代码块
        # 3. 退出时：调用__exit__()，停止计时并输出结果
    
    等效的手动实现：
    # 手动计时（繁琐）
    timer = common.Timer()
    for i in range(1000): 
        net(x)
    elapsed = timer.stop()
    print(f'无torchscript: {elapsed:.4f} sec')
    
    # 使用Benchmark类（简洁）
    with Benchmark('无torchscript'):
        for i in range(1000): net(x)
    '''
    net = get_net() # 获取原始模型（动态图）
    with common.Benchmark('无torchscript'):
        for i in range(1000): net(x) # 执行1000次推理
    '''执行过程：
    # 每次循环都发生：
    1. Python调用net.forward() → 2. 逐层执行 → 3. 返回结果
       ↓
    Python解释器开销 × 1000次！
    '''

    net = torch.jit.script(net) # 编译为TorchScript
    with common.Benchmark('有torchscript'):
        for i in range(1000): net(x) # 执行1000次推理（编译后）
    '''执行过程：
    # 编译后执行：
    1. 整个计算图已优化为原生代码
    2. 直接调用编译后的函数（无Python逐层调用）
    3. 高效执行
    '''

    ''' 等价于PyTorch标准写法：
    torch.save(net.state_dict(), 'my_mlp.pth')  # 保存模型参数
    # 或
    torch.save(net, 'my_mlp.pth')  # 保存整个模型
    '''
    net.save('my_mlp') # 保存模型：将训练好的神经网络模型保存到文件
    '''
    # !ls -lh my_mlp*  列出所有以 my_mlp开头的文件，显示详细信息
    命令分解：
    !   ：在Jupyter Notebook中执行shell命令
    ls  ：列出文件（Linux/macOS命令）
    -lh ：参数组合
        -l：详细列表格式
        -h：人类可读的文件大小（KB/MB/GB）
    my_mlp*：匹配所有以my_mlp开头的文件
    Windows等价命令： 
        !dir my_mlp*  # Windows系统
    '''
    from pathlib import Path
    # 使用pathlib查看结果
    model_path = Path('my_mlp')
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        print(f"模型文件: {model_path.name}")
        print(f"文件大小: {size_kb:.1f} KB")
        print(f"完整路径: {model_path.absolute()}")
    else:
        print("❌ 文件不存在，保存失败！")
# learn_Compilers_and_interpreters()


def learn_asynchronous_computing():
    ''' 异步计算：GPU性能对比测试
    展示计算的性能差异：PyTorch GPU计算 vs NumPy CPU '''
    # GPU计算热身
    # 作用：避免冷启动开销（第一次GPU操作较慢）
    device = common.try_gpu()  # 自动检测GPU，若无则使用CPU
    a = torch.randn(size=(1000, 1000), device=device)  # 在GPU上创建随机矩阵
    b = torch.mm(a, a)  # GPU矩阵乘法（预热，避免首次运行开销）

    # 1、NumPy的CPU计算性能
    # 预期结果：相对较慢，因为CPU处理大矩阵乘法效率较低
    with common.Benchmark('numpy (NumPy CPU基准测试)'):
        for _ in range(10):  # 执行10次
            a = numpy.random.normal(size=(1000, 1000))  # CPU生成随机矩阵
            b = numpy.dot(a, a)  # CPU矩阵乘法

    # 2、PyTorch的GPU异步计算性能
    # 关键点：没有调用torch.cuda.synchronize() → 测量的是排队时间而非实际计算时间
    with common.Benchmark('torch (PyTorch GPU异步测试)'):
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)

    # 3、GPU同步计算的性能
    # 关键点：显式同步 → 测量实际计算时间
    with common.Benchmark('torch (PyTorch GPU同步测试)'):  # 默认描述
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)
        # 若安装的PyTorch是cpu版本，则没有CUDA支持，无法使用GPU。则下句需注释
        torch.cuda.synchronize(device)  # 等待所有GPU操作完成

    # 验证GPU计算正确性
    # 预期输出：tensor([[3., 3.]], device='cuda:0')
    x = torch.ones((1, 2), device=device)
    y = torch.ones((1, 2), device=device)
    z = x * y + 2  # 简单GPU计算
    print(f"{z}")  # 输出结果，验证GPU工作正常
# learn_asynchronous_computing()


def learn_automatic_parallelism():
    '''自动并行'''
    devices = common.try_all_gpus()
    print(f"检测到的总设备数量：{len(devices)}")
    def run(x):
        return [x.mm(x) for _ in range(50)]

    x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
    x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])

    run(x_gpu1)
    run(x_gpu2)  # 预热设备
    torch.cuda.synchronize(devices[0])
    torch.cuda.synchronize(devices[1])

    with common.Benchmark('GPU1 time'):
        run(x_gpu1) # 在GPU1上执行计算
        torch.cuda.synchronize(devices[0]) # 等待GPU1完成

    with common.Benchmark('GPU2 time'):
        run(x_gpu2) # 在GPU2上执行计算
        torch.cuda.synchronize(devices[1]) # 等待GPU2完成

    with common.Benchmark('GPU1 & GPU2'):
        run(x_gpu1)  # 启动GPU1计算
        run(x_gpu2)  # 立即启动GPU2计算 ← 关键并行点
        torch.cuda.synchronize() # 等待所有GPU操作完成

    # 并行计算与通信
    def copy_to_cpu(x, non_blocking=False):
        """将GPU上的张量复制到CPU
        x: 可以是单个张量或张量列表
        non_blocking: 是否使用非阻塞传输（异步）
            # 阻塞模式（默认）
            y.to('cpu') # 必须等待复制完成才能继续执行
            # 非阻塞模式
            y.to('cpu', non_blocking=True) # 启动传输后立即返回，允许后续计算继续
        返回：CPU上的张量或张量列表
        """
        # 如果是列表/元组，递归处理每个元素
        return [y.to('cpu', non_blocking=non_blocking) for y in x]

    # 基准测试1：纯GPU计算
    with common.Benchmark('在GPU1上运行'):
        # 在GPU1上执行计算任务（假设run()是某个计算密集型函数）
        y = run(x_gpu1)
        # 同步GPU流，确保计算完成（阻塞等待）
        torch.cuda.synchronize()
        """
        这个测试测量：
        - 纯GPU计算时间
        - 不涉及数据传输
        """

    # 基准测试2：纯数据传输
    with common.Benchmark('复制到CPU'):
        # 将GPU计算结果复制到CPU（默认阻塞模式）
        y_cpu = copy_to_cpu(y)
        # 同步确保传输完成
        torch.cuda.synchronize()
        """
        这个测试测量：
        - 从GPU到CPU的数据传输时间
        - 阻塞模式：计算和传输是串行的
        """

    # 基准测试3：计算与传输重叠
    with common.Benchmark('在GPU1上运行并复制到CPU'):
        # 执行GPU计算
        y = run(x_gpu1)
        # 使用非阻塞模式将数据复制到CPU
        y_cpu = copy_to_cpu(y, True)
        # 同步确保所有操作完成
        torch.cuda.synchronize()
        """
        这个测试测量：
        - 计算与传输的重叠执行时间
        - non_blocking=True允许：
          1. GPU计算继续进行
          2. 同时将数据从GPU复制到CPU
        关键点：
        - 总时间 ≈ max(计算时间, 传输时间)
        - 而非计算时间+传输时间
        """
# learn_automatic_parallelism()



# 初始化模型参数
scale = 0.01 # 控制参数初始化的缩放因子，防止梯度爆炸

# 卷积层1参数：输入通道1，输出通道20，3x3卷积核
W1 = torch.randn(size=(20, 1, 3, 3)) * scale  # 权重（随机初始化）
b1 = torch.zeros(20)                          # 偏置（初始为0）

# 卷积层2参数：输入通道20，输出通道50，5x5卷积核
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)

# 全连接层1参数：800维输入，128维输出
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)

# 全连接层2参数：128维输入，10维输出（对应10分类）
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4] # 所有参数列表


# 定义模型
def lenet(X, params):
    """LeNet-5模型实现（手动参数管理版本）
    X: 输入张量 (batch_size, 1, 28, 28)
    params: 包含所有参数的列表 [W1,b1,W2,b2,W3,b3,W4,b4]
    """
    # 第一层：卷积 → ReLU → 平均池化
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])  # 卷积
    h1_activation = F.relu(h1_conv)                                # 激活
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2)) # 池化

    # 第二层：卷积 → ReLU → 平均池化
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))

    # 展平后进入全连接层
    h2 = h2.reshape(h2.shape[0], -1)  # (batch_size, 50 * 4 * 4=800)

    # 全连接层1：线性变换 → ReLU
    h3_linear = torch.mm(h2, params[4]) + params[5] # 矩阵乘法
    h3 = F.relu(h3_linear)

    # 输出层：线性变换（无激活函数）
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数（reduction='none'表示不自动求平均/求和）
loss = nn.CrossEntropyLoss(reduction='none')

def get_params(params, device):
    """将参数移动到指定设备并启用梯度计算
    params: 参数列表
    device: 目标设备（如'cuda:0'）
    """
    new_params = [p.to(device) for p in params] # 移动到设备
    for p in new_params:
        p.requires_grad_() # 启用梯度跟踪
    return new_params

# 示例：将参数移动到GPU 0
new_params = get_params(params, common.try_gpu(0))
print('b1 权重:', new_params[1])      # 打印偏置参数
print('b1 梯度:', new_params[1].grad) # 打印梯度（初始为None）

def allreduce(data):
    """多GPU梯度聚合（简化版实现）
    data: 包含各GPU上张量的列表 [tensor_gpu0, tensor_gpu1, ...]
    """
    # 梯度求和：将所有GPU的数据累加到GPU 0
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device) # 注意[:]原地操作

    # 广播结果：将GPU 0的结果复制回其他GPU
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

# 测试AllReduce
data = [torch.ones((1, 2), device=common.try_gpu(i)) * (i + 1) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
"""
输出示例（假设有2个GPU）：
tensor([[1., 1.]], device='cuda:0') 
tensor([[2., 2.]], device='cuda:1')
"""
allreduce(data)
print('allreduce之后：\n', data[0], '\n', data[1])
"""
输出示例：
tensor([[3., 3.]], device='cuda:0')  # 1+2=3
tensor([[3., 3.]], device='cuda:1')  # 从GPU0同步
"""







