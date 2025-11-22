import torch
from torch import nn
import common


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
learn_Compilers_and_interpreters()







