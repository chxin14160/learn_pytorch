import numpy as np
import torch
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import common


# 下载器与数据集配置
# 为 time_machine 数据集注册下载信息，包括文件路径和校验哈希值（用于验证文件完整性）
downloader = common.C_Downloader()
DATA_HUB = downloader.DATA_HUB  # 字典，存储数据集名称与下载信息
DATA_URL = downloader.DATA_URL  # 基础URL，指向数据集的存储位置

DATA_HUB['airfoil'] = (DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')



def learn_optimization_and_deep_learning():
    '''优化和深度学习'''
    def f(x):
        '''风险函数
        理论风险值：x * cos(πx)
        理论风险的简化模拟：
            模拟真实数据分布下的期望损失
            cos(πx)项：反映损失随 参数x 的振荡变化
            全局最小值对应最优模型参数
        '''
        return x * torch.cos(np.pi * x)

    def g(x):
        '''经验风险函数
        经验风险值 = 风险函数 + 高频噪声项
        f(x)：继承理论风险的基础形态
        0.2*cos(5πx)：模拟有限样本导致的波动
            振幅0.2：反映 数据量不足带来的估计误差
            高频5π ：反映 经验风险的局部波动性
        '''
        # 在风险函数基础上添加高频振荡项（模拟训练数据噪声）
        return f(x) + 0.2 * torch.cos(5 * np.pi * x)

    def visual_risk_functions_comparison():
        '''可视化：风险函数对照
        即 理论风险函数f(x) 与 经验风险函数g(x) 的差异图'''
        # 绘制对比图：风险函数 & 经验风险函数
        x = torch.arange(0.5, 1.5, 0.01) # 参数搜索范围
        fig1, ax1 = common.plot(x, [f(x), g(x)], 'x', 'risk',
                                legend=['Risk (f)', 'Empirical Risk (g)'], # 添加图例
                                xlim=(0.47, 1.53), ylim=(-1.3, 0.25),
                                figsize=(6, 4), title='Risk Functions Comparison',
                                show_internally = False)
        # 添加关键点注释
        common.annotate('min of\nempirical risk',  (1.0, -1.2), (0.7, -0.87), ax=ax1) # 经验风险最小值点
        common.annotate('min of risk', (1.1, -1.05), (1.05, -0.5), ax=ax1,) # 理论风险最小值点
    # visual_risk_functions_comparison()

    def test_local_minimum():
        ''' 局部最小值 '''
        # 绘制对比图：局部最小值 & 全局最小值
        x = torch.arange(-1.0, 2.0, 0.01)
        # 使用前面的风险函数 f(x)，对比两种风险时范围是0.5~1.5，现在看最小值的范围的-1.0~2.0
        fig2, ax2 = common.plot(x, [f(x), ], 'x', 'f(x)',
                                xlim=(-1.05, 2.05), ylim=(-1.5, 3),
                                title='Local vs Global Minimum',
                                show_internally = False)
        common.annotate('local minimum', (-0.3, -0.25), (-0.37, 1.5), ax=ax2)
        common.annotate('global minimum', (1.1, -0.95), (0.9, 0.8), ax=ax2)
    # test_local_minimum()

    def visual_saddle_pnt():
        ''' 可视化：鞍点 '''
        # 绘制鞍点示例图
        x = torch.arange(-2.0, 2.0, 0.01) # 扩展参数范围
        fig3, ax3 = common.plot(x, [x**3], 'x', 'f(x)',
                              xlim=(-2, 2), ylim=(-8, 8),
                              figsize=(6, 4), title='Saddle Point Example',
                                show_internally = False)
        common.annotate('saddle point', (0, -0.2), (-0.52, -5.0),
                        ax = ax3, fontsize = 12,
                        arrowprops = dict(arrowstyle='fancy', color='green')) # 鞍点注释
        # 添加参考线
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)  # 添加x轴虚线
        plt.axvline(x=0, color='k', linestyle='--', linewidth=0.8)  # 添加y轴虚线


        # 创建二维网格：在[-1,1]区间生成101x101的网格点坐标矩阵
        # torch.linspace创建等差数列，torch.meshgrid将一维坐标扩展为二维网格
        x, y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 101),  # x坐标范围[-1,1]，101个点
            torch.linspace(-1.0, 1.0, 101))   # y坐标范围[-1,1]，101个点

        # 定义鞍点函数：z = x² - y²
        # 该函数在(0,0)处Hessian矩阵特征值为(2, -2)，是典型的鞍点
        z = x**2 - y**2

        ax = plt.figure().add_subplot(111, projection='3d') # 创建三维坐标轴对象

        # 绘制三维线框图
        # plot_wireframe以线框形式绘制曲面
        # 通过rstride/cstride控制行/列方向的网格密度（步长）
        ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})

        # 红色叉号标记鞍点位置(0,0,0)，直观展示梯度消失点
        ax.plot([0], [0], [0], 'rx') # 第一个列表表示x坐标，第二个y坐标，第三个z坐标

        # 设置坐标轴刻度
        ticks = [-1, 0, 1]      # 统一设置刻度为[-1,0,1]
        plt.xticks(ticks)       # 设置x轴刻度
        plt.yticks(ticks)       # 设置y轴刻度
        ax.set_zticks(ticks)    # 设置z轴刻度

        # 添加坐标轴标签
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlabel('z')  # 补充z轴标签
        # plt.savefig('saddle_point.png', dpi=300, bbox_inches='tight') # 保存图像到文件
    # visual_saddle_pnt()

    # 梯度消失示例
    x = torch.arange(-2.0, 5.0, 0.01)
    fig4, ax4 = common.plot(x, [torch.tanh(x)], 'x', 'f(x)', show_internally=False)
    common.annotate('vanishing gradient', (4, 1), (2, -0.20), ax=ax4)
# learn_optimization_and_deep_learning()


def learn_convexity():
    '''凸性，演示：凸函数与非凸函数，局部极小值是全局极小值'''
    # 定义三个函数
    f = lambda x: 0.5 * x**2            # 凸函数  （典型，开口向上的抛物线）
    g = lambda x: torch.cos(np.pi * x)  # 非凸函数（余弦波）
    h = lambda x: torch.exp(0.5 * x)    # 凸函数  （指数增长）

    # 生成数据
    x = torch.arange(-2, 2, 0.01)
    segment = torch.tensor([-1.5, 1]) # 用于绘制线段的两个端点

    # d2l.use_svg_display()
    _, axes = plt.subplots(1, 3, figsize=(9, 3))

    for ax, func, name in zip(axes, [f, g, h],
                              ['f(x) = 0.5x²', 'g(x) = cos(πx)', 'h(x) = exp(0.5x)']):
        _, ax = common.plot([x, segment], [func(x), func(segment)], axes=ax,
                            title = name, show_internally=False)


    f = lambda x: (x - 1) ** 2
    common.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
# learn_convexity()


def learn_gradient_descent():
    '''梯度下降：一维、多元、牛顿法'''
    def visual_1d_gradient_descent():
        '''一维梯度下降，及对应可视化'''
        def f(x):  # 目标函数 f(x) = x²
            return x ** 2

        def f_grad(x):  # 目标函数的梯度(导数) f'(x) = 2x
            return 2 * x

        def gd(eta, f_grad):
            x = 10.0 # 初始值
            results = [x]
            for i in range(10): # 迭代10次
                x -= eta * f_grad(x)
                results.append(float(x))
                print(f"epoch ：{i+1}，当前x={x:.7f}")
            print(f'epoch 10, x: {x:f}')
            return results

        results = gd(0.2, f_grad) # 传入0.2的学习率(步长大小)，和对应导数(坡度)
        common.show_trace(results, f) # 显示梯度下降轨迹
        common.show_trace(gd(0.05, f_grad), f) # 学习率改为0.05，可视化梯度下降轨迹
        common.show_trace(gd(1.1, f_grad), f) # 学习率改为1.1，可视化梯度下降轨迹


        c = torch.tensor(0.15 * np.pi) # 预定义的常数 c = 0.15π ≈ 0.4712

        def f(x):  # 目标函数 f(x) = x * cos(c * x)
            return x * torch.cos(c * x)

        def f_grad(x):  # 目标函数的梯度 f'(x) = cos(c*x) - c*x*sin(c*x)
            return torch.cos(c * x) - c * x * torch.sin(c * x)

        common.show_trace(gd(2, f_grad), f) # 学习率设为2
    # visual_1d_gradient_descent()

    def visual_2d_gradient_descent():
        '''多元梯度下降，及对应可视化'''
        def f_2d(x1, x2):  # 目标函数 f(x) = x₁² + 2x₂²
            return x1 ** 2 + 2 * x2 ** 2

        def f_2d_grad(x1, x2):  # 目标函数的梯度：[∂f/∂x1, ∂f/∂x2] = [2x1, 4x2]
            return (2 * x1, 4 * x2)

        def gd_2d(x1, x2, s1, s2, f_grad):
            """标准的梯度下降更新规则
            x1, x2: 当前参数值
            s1, s2: 状态变量（在此简单GD中未使用）
            f_grad: 梯度计算函数
            返回: 更新后的参数和状态 (new_x1, new_x2, 0, 0)
            """
            g1, g2 = f_grad(x1, x2) # 计算梯度
            # 梯度下降更新：x ← x - η * ∇f(x)
            new_x1 = x1 - eta * g1
            new_x2 = x2 - eta * g2
            return (new_x1, new_x2, 0, 0)  # 返回新参数，状态重置为0

        eta = 0.1 # 学习率
        # 执行梯度下降并可视化结果
        results = common.train_2d(gd_2d, f_grad=f_2d_grad)
        common.show_trace_2d(f_2d, results)
    # visual_2d_gradient_descent()

    def visual_newton():
        '''牛顿法演示'''
        c = torch.tensor(0.5) # 常数系数

        def f(x):  # O目标函数：双曲余弦函数，形状类似开口向上的抛物线
            return torch.cosh(c * x)

        def f_grad(x):  # 目标函数的梯度（一阶导数）
            # f'(x) = d/dx [cosh(c*x)] = c * sinh(c*x)
            return c * torch.sinh(c * x) # 双曲正弦函数是双曲余弦的导数

        def f_hess(x):  # 目标函数的Hessian（二阶导数，即 Hessian标量）
            # f''(x) = d/dx [c * sinh(c*x)] = c² * cosh(c*x)
            return c**2 * torch.cosh(c * x) # 双曲余弦的二阶导仍是双曲余弦

        def newton(eta=1):
            """牛顿法优化器
            eta(η)：控制更新步长的缩放因子
                    通常设为1，不稳定时可降低0.1-0.5
            """
            x = 10.0  # 初始点
            results = [x]  # 记录轨迹
            for i in range(10):
                # 牛顿法更新公式：x ← x - η * f'(x)/f''(x)
                ''' 牛顿法更新公式解析
                数学形式：x_{k+1} = x_k - η * [∇²f(x_k)]⁻¹ * ∇f(x_k)
                对于单变量情况：
                [∇²f(x_k)]⁻¹简化为 1/f''(x_k)
                因此更新公式简化为 x_k - f'(x_k)/f''(x_k)
                '''
                x -= eta * f_grad(x) / f_hess(x)
                results.append(float(x))
            print('epoch 10, x:', x)
            return results

        common.show_trace(newton(), f)


        c = torch.tensor(0.15 * np.pi) # 常数系数

        def f(x):  # 目标函数：振荡函数
            # c：控制函数振荡频率(值越大函数振荡越剧烈)
            return x * torch.cos(c * x) # 产生周期性振荡的函数

        def f_grad(x):
            ''' 目标函数的梯度（一阶导数）
            目标函数：f(x) = x * cos(c*x)
            一阶导数（乘积法则）：
            f'(x) = d/dx [x] * cos(c*x) + x * d/dx [cos(c*x)]
                  = 1 * cos(c*x) + x * (-c * sin(c*x))
                  = cos(c*x) - c*x*sin(c*x)
            '''
            return torch.cos(c * x) - c * x * torch.sin(c * x)

        def f_hess(x):
            ''' 目标函数的Hessian（二阶导数）
            目标函数：f(x) = x * cos(c*x)
            二阶导数：
            f''(x) = d/dx [cos(c*x)] - c * d/dx [x*sin(c*x)]
                   = -c*sin(c*x) - c*[sin(c*x) + x*c*cos(c*x)]
                   = -2c*sin(c*x) - c²x*cos(c*x)
            '''
            return - 2 * c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)

        common.show_trace(newton(), f)      # 学习率为1.0
        common.show_trace(newton(0.5), f)   # 学习率为0.5
    # visual_newton()
# learn_gradient_descent()


# 初始化全局变量
t = 1 # 时间步计数器（从1开始）

def learn_sgd():
    '''随机梯度下降：随机梯度更新 & 指数衰减和多项式衰减演示'''
    def f(x1, x2):  # 目标函数 f(x) = x₁² + 2x₂²
        return x1 ** 2 + 2 * x2 ** 2

    def f_grad(x1, x2):  # 目标函数的梯度：[∂f/∂x1, ∂f/∂x2] = [2x1, 4x2]
        return 2 * x1, 4 * x2

    def sgd(x1, x2, s1, s2, f_grad):
        """ 随机梯度下降更新
        x1, x2: 当前参数值
        s1, s2: 状态变量（为动量法等预留的，此处未使用）
        f_grad: 梯度计算函数
        返回: 更新后的参数和状态 (new_x1, new_x2, 0, 0)
        """
        g1, g2 = f_grad(x1, x2) # 计算真实梯度
        # 添加均值为0、标准差为1的高斯噪声
        # 模拟有噪声的梯度（添加随机噪声，模拟SGD的批次采样中批次梯度的不确定性）
        g1 += torch.normal(0.0, 1, (1,)).item() # 给x1梯度加噪声
        g2 += torch.normal(0.0, 1, (1,)).item() # 给x2梯度加噪声
        eta_t = eta * lr() # 计算当前步长（此处学习率固定，固定为 0.1 * 1 = 0.1）
        # 参数更新：x ← x - η * (∇f(x) + noise)，即带噪声的梯度下降
        return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0) # 返回新参数和清零状态

    def constant_lr():
        return 1 # 固定学习率系数

    eta = 0.1         # 基础学习率
    lr = constant_lr  # 学习率函数（此处为常数）
    common.show_trace_2d(f, common.train_2d(sgd, steps=50, f_grad=f_grad))

    global t # 因为将这一章节的代码都封装了起来，所以这里需再声明一次t是全局变量

    def exponential_lr():
        ''' 指数学习率衰减策略
        学习率按 η = e^(-0.1*t)衰减
        调整指数系数（如-0.1）控制衰减速度
        衰减特点：
            初期下降较快（前几步学习率迅速减小）
            后期趋于平缓（当t较大时，e^(-0.1*t)接近0）
        适用场景：需要快速降低学习率的任务，但可能过早失去学习能力
        '''
        # 在函数外部定义，而在内部更新的全局变量
        global t  # 声明使用全局变量t
        t += 1    # 每次调用递增步数计数器
        return math.exp(-0.1 * t)  # 指数衰减公式: η = e^(-0.1*t)

    # 初始化全局变量
    t = 1                # 时间步计数器（从1开始）
    lr = exponential_lr  # 设置学习率函数为指数衰减
    common.show_trace_2d(f, common.train_2d(sgd, steps=1000, f_grad=f_grad))


    def polynomial_lr():
        ''' 多项式学习率衰减策略
        学习率按 η = (1 + 0.1*t)^(-0.5)衰减
        优先尝试 (1 + α*t)^(-0.5)，调整α
        衰减特点：
            初期下降平缓（保护早期快速学习能力）
            后期持续稳定衰减（避免后期震荡）
        理论保障：对于凸优化问题可证明收敛性
        参数选择：
            分母中的 0.1控制衰减速度
            指数 -0.5是理论推荐值
        '''
        # 在函数外部定义，而在内部更新的全局变量
        global t  # 声明使用全局变量t
        t += 1    # 每次调用递增步数计数器
        return (1 + 0.1 * t) ** (-0.5)  # 多项式衰减公式: η = (1+0.1*t)^(-0.5)

    # 重新初始化全局变量
    t = 1               # 重置时间步计数器（因为切换策略了）
    lr = polynomial_lr  # 设置学习率函数为多项式衰减
    common.show_trace_2d(f, common.train_2d(sgd, steps=50, f_grad=f_grad)) # 迭代次数降到50
# learn_sgd()


def learn_MiniBatch_sgd():
    '''小批量随机梯度下降'''
    def test_vectorization_and_cache_and_miniBatch():
        '''演示：向量化和缓存，小批量'''
        timer = common.Timer()      # 初始化计时器
        A = torch.zeros(256, 256)   # A: 256x256零矩阵（用于存储结果）
        B = torch.randn(256, 256)   # B: 256x256随机矩阵
        C = torch.randn(256, 256)   # C: 256x256随机矩阵

        # 测试一：【逐元素】计算A=BC
        timer.start()   # 开始计时
        for i in range(256):      # 遍历行
            for j in range(256):  # 遍历列
                # 计算 B的第i行 与 C的第j列 的点积
                A[i, j] = torch.dot(B[i, :], C[:, j])
        stop_time = timer.stop()    # 停止计时
        print(f"【逐元素】计算耗时：{stop_time}")


        # 测试二：【逐列】计算A=BC
        timer.start()
        for j in range(256):  # 仅遍历列
            # 计算B与C的第j列的矩阵-向量乘积
            A[:, j] = torch.mv(B, C[:, j])  # mv = matrix-vector multiplication
        stop_time = timer.stop()
        print(f"【逐列】计算耗时：{stop_time}")


        # 测试三：【一次性】计算A=BC （一次性完整矩阵乘法）
        timer.start()
        # 直接调用优化后的矩阵乘法
        A = torch.mm(B, C)  # mm = matrix-matrix multiplication
        stop_time = timer.stop()
        print(f"【一次性完整矩阵乘法】计算耗时：{stop_time}")


        min_time = 1e-6  # 1微秒下限(避免除以零，增加最小时间阈值)

        # 乘法和加法作为单独的操作（在实践中融合）即
        # 底层计算库(如BLAS、cuBLAS)会将矩阵乘法(GEMM) 和
        # 后续的加法操作（如偏置项相加）合并为一个复合操作，从而显著提升计算效率
        gigaflops = [2/max(i, min_time) for i in timer.times]
        print(f'performance in Gigaflops 性能对比（GFLOPS）: \n'
              f'element 逐元素计算: {gigaflops[0]:.3f}, \n'
              f'column 逐列计算 : {gigaflops[1]:.3f}, \n'
              f'full 完整矩阵乘法: {gigaflops[2]:.3f}\n')


        timer.start()
        for j in range(0, 256, 64): # 一次性分为64列的“小批量”
            A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
        timer.stop()
        print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
    # test_vectorization_and_cache_and_miniBatch()



    def sgd(params, states, hyperparams):
        '''sgd优化器
        params：需要被优化的变量列表（模型参数列表）
        states：状态
        hyperparams：存放超参数的字典
        '''
        for p in params:
            # .sub_() 原地减法操作，直接修改参数值
            # 等效于 p.data = p.data - η·∇L，但更高效
            p.data.sub_(hyperparams['lr'] * p.grad) # 参数更新
            p.grad.data.zero_() # 梯度清零

    # 训练流程封装
    def train_sgd(lr, batch_size, num_epochs=2):
        '''训练流程封装：入口函数
        1、初始化训练参数：如学习率，批量大小，训练轮数，所用优化器(sgd)
        2、启动训练流程
        '''
        # 获取数据迭代器和特征维度
        data_iter, feature_dim = common.get_data_ch11(downloader, batch_size)
        # 启动训练流程
        return common.train_ch11(
            sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

    # 执行训练：
    ''' 全批量梯度下降(GD)，每1500样本更新一次
    lr=1：大学习率（全批量梯度更稳定，允许大学习率）
    batch_size=1500 ：全批量（使用全部数据计算梯度）
    num_epochs=10   ：训练10轮
    行为：
        每轮（epoch）计算所有1500个样本的平均梯度，更新一次参数。
        共更新 10次（每轮1次）
    '''
    # 学习率1，批量大小1500，训练轮数10
    gd_res = train_sgd(1, 1500, 10)

    ''' 随机梯度下降(SGD)，每个样本更新一次
    lr=0.005：极小学习率（单样本梯度噪声大，需小步长）
    batch_size=1：纯SGD（每个样本单独更新）
    num_epochs=2（默认值）
    行为：
        每轮（epoch）遍历1500个样本，每个样本更新一次参数。
        共更新 1500 × 2 = 3000次（每轮1500次，迭代次数默认=2）
    '''
    sgd_res = train_sgd(0.005, 1)

    ''' 小批量梯度下降（Mini-batch）
    每轮更新次数 分别为：
    15次/轮（1500/100）
    150次/轮（1500/10）
    但实际只更新了默认2轮(为了快速验证)，所以实际总更新次数如下：
    15 × 2 = 30
    150 × 2 = 300
    '''
    mini1_res = train_sgd(.4, 100) # 中等批量
    mini2_res = train_sgd(.05, 10) # 小批量

    # zip() 将时间序列和损失序列分离。将所有 时间序列和损失序列 分别整合成元组
    #       元组内每个元素为 某种梯度下降法的 时间序列 或 损失序列
    # map(list, ...) 将zip生成的元组转换为列表(将 转列表 应用到每个zip生成的元组上)
    # *list() 将嵌套列表解包为独立参数(相当于4种梯度下降法的 时间序列/损失序列 都单独拎出来，而不是大的时间序列整体)
    common.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
             'time (sec)', 'loss',
                xlim=[1e-2, 10], # 时间轴范围（0.01~10秒）
                xscale='log',    # 时间轴用对数坐标（便于观察初期快速下降）
                figsize=[6, 3],
                legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
    # plt.gca().set_xscale('log')


    # 获取数据迭代器，批量大小10(选用小批量训练)
    data_iter, _ = common.get_data_ch11(downloader, batch_size=10)
    trainer = torch.optim.SGD # 指定优化器为PyTorch内置的SGD（随机梯度下降）
    common.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
# learn_MiniBatch_sgd()


def learn_momentum():
    '''动量法'''
    def test_momentum_method_effectiveness_demonstration():
        '''动量法效果演示'''
        eta = 0.4 # 学习率
        def f_2d(x1, x2):
            """二维目标函数：f(x1, x2) = 0.1*x1² + 2*x2²
            椭圆抛物面，最小值在原点(0,0)
            在x2方向（系数2）比x1方向（系数0.1）陡峭得多
            """
            return 0.1 * x1 ** 2 + 2 * x2 ** 2
        def gd_2d(x1, x2, s1, s2):
            """自定义的梯度下降更新函数（注意：这不是标准形式！）
            x1, x2: 当前参数值
            s1, s2: 预留的状态变量（此处未使用）
            返回：更新后的参数 (x1_new, x2_new, 0, 0)

            注意：这里的更新规则很特殊！
                标准梯度下降应为：x = x - η * ∇f(x)
                其中 ∇f(x) = [∂f/∂x1, ∂f/∂x2] = [0.2*x1, 4*x2]

            此实现直接写为：x1_new = x1 - η*0.2*x1 = (1 - 0.2η)*x1
                         x2_new = x2 - η* 4 *x2 = (1 - 4η)*x2
            """
            return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

        common.show_trace_2d(f_2d, common.train_2d(gd_2d))

        eta = 0.6 # 学习率略微提高到 0.6
        common.show_trace_2d(f_2d, common.train_2d(gd_2d))


        def momentum_2d(x1, x2, v1, v2):
            """动量法更新函数
            x1, x2: 当前参数值
            v1, v2: 动量状态（历史梯度累积）
            返回：更新后的参数和动量状态
            """
            # 动量更新(动量累积)：v_new = β*v_old + 当前梯度
            v1 = beta * v1 + 0.2 * x1  # x1方向：梯度 = ∂f/∂x1 = 0.2*x1
            v2 = beta * v2 + 4 * x2    # x2方向：梯度 = ∂f/∂x2 = 4*x2

            # 参数更新：x_new = x_old - η*v_new
            return x1 - eta * v1, x2 - eta * v2, v1, v2

        eta, beta = 0.6, 0.5 # 学习率依旧是0.6，动量系数0.5（保留50%的历史动量）
        common.show_trace_2d(f_2d, common.train_2d(momentum_2d))

        eta, beta = 0.6, 0.25 # 动量系数减半至0.25（只保留25%的历史动量）
        common.show_trace_2d(f_2d, common.train_2d(momentum_2d))


        betas = [0.95, 0.9, 0.6, 0] # 动量系数列表：从强记忆到无记忆
        '''
        纵轴含义：beta ** x
        表示t步前的梯度在当前动量项中的权重系数
        例如：beta=0.9时，10步前的梯度权重是 0.9^10 ≈ 0.35
        '''
        for beta in betas: # 对每个beta值绘制衰减曲线
            # 创建时间序列：0到39（代表过去的时间步）
            x = torch.arange(40).detach().numpy()  # [0, 1, 2, ..., 39]
            # 计算β^t：表示t步前的梯度在当前动量中的权重
            plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
        plt.xlabel('time')  # x轴：时间（过去的步数）
        plt.legend()        # 显示图例
    # test_momentum_method_effectiveness_demonstration()

    # 获取数据集的迭代器和特征维度，批量大小为10
    data_iter, feature_dim = common.get_data_ch11(downloader, batch_size=10)

    def momentum_method_StartFromScratch():
        '''动量法：从零开始实现'''
        def init_momentum_states(feature_dim):
            ''' 动量状态初始化
            根据特征维度，初始化动量法的状态参数
            为每个可训练参数创建对应的动量变量v，初始化为0
            '''
            v_w = torch.zeros((feature_dim, 1)) # 权重w的动量状态（形状同w）
            v_b = torch.zeros(1)                # 偏置b的动量状态（形状同b）
            return (v_w, v_b)

        def sgd_momentum(params, states, hyperparams):
            '''动量法优化器'''
            for p, v in zip(params, states): # 同时遍历参数和对应的动量状态
                with torch.no_grad():        # 禁用梯度跟踪（纯数值计算）
                    # 动量更新：v_new = β * v_old + 当前梯度
                    v[:] = hyperparams['momentum'] * v + p.grad # 原地更新动量状态（保持内存引用）
                    # 参数更新：p_new = p_old - η * v_new
                    p[:] -= hyperparams['lr'] * v # 原地更新参数值
                p.grad.data.zero_()           # 清零梯度，准备下一轮计算（防止梯度累积）

        def train_momentum(lr, momentum, num_epochs=2):
            '''训练函数封装'''
            common.train_ch11(
                sgd_momentum,                    # 优化器函数
                init_momentum_states(feature_dim), # 初始化动量状态
                {'lr': lr, 'momentum': momentum}, # 超参数字典
                data_iter,                        # 数据迭代器
                feature_dim,                      # 特征维度
                num_epochs                        # 训练轮数
            )

        # # 获取数据集的迭代器和特征维度，批量大小为10
        # data_iter, feature_dim = common.get_data_ch11(downloader, batch_size=10)

        # 中等动量，标准学习率。效果：平衡收敛速度与稳定性
        train_momentum(0.02, 0.5) # 学习率0.02，动量系数0.5

        # 强动量，降低学习率（避免震荡）。效果：更强记忆效应，需更小心控制步长
        # 动量系数增加到0.9(相当于有效样本数量增加)，学习率略微降至0.01以确保可控
        train_momentum(0.01, 0.9)

        # 强动量，更小学习率（确保收敛）。效果：最稳定收敛，但可能稍慢
        # 学习率再降至0.005，会产生良好的收敛性能
        train_momentum(0.005, 0.9)
    # momentum_method_StartFromScratch()

    def momentum_method_SimpleImplementation():
        '''动量法：简洁实现'''
        trainer = torch.optim.SGD # 指定优化器类 (注意：是类引用，而非示例)
        # PyTorch的torch.optim.SGD根据是否提供momentum参数，自动选择工作模式
        # 这里提供了momentum参数，所以自动启用了动量法
        # 内部实现：v = momentum * v + grad; p = p - lr * v
        common.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
    # momentum_method_SimpleImplementation()


    # 可视化不同曲率（λ）下，梯度下降的收敛行为
    # 定义不同的曲率值（控制损失函数的陡峭程度）
    lambdas = [0.1, 1, 10, 19]  # λ值：从平缓到陡峭
    eta = 0.1 # 固定学习率为0.1
    plt.figure(figsize=(6, 4))  # 创建图形
    for lam in lambdas:         # 对每个λ值绘制收敛曲线
        t = torch.arange(20).detach().numpy() # 创建时间序列：0到19（20个时间步）
        # 计算每个时间步的收敛因子：(1 - ηλ)^t
        plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
    plt.xlabel('time')  # x轴：时间（迭代次数）
    plt.legend()        # 显示图例
# learn_momentum()


def learn_AdaGrad_algorithm():
    '''AdaGrad算法'''
    def adagrad_2d(x1, x2, s1, s2):
        '''AdaGrad优化器实现（二维版本）
        x1, x2: 当前参数值
        s1, s2: 梯度平方累积状态（历史记忆）
        '''
        eps = 1e-6  # 小常数，防止除以零

        # 1. 计算当前梯度（目标函数f(x1,x2)=0.1*x1²+2*x2²的梯度）
        g1 = 0.2 * x1  # ∂f/∂x1 = 0.2*x1
        g2 = 4 * x2    # ∂f/∂x2 = 4*x2

        # 2. 累积梯度平方（AdaGrad核心：记忆历史梯度大小）
        s1 += g1 ** 2  # 累积x1方向的梯度平方
        s2 += g2 ** 2  # 累积x2方向的梯度平方

        # 3. AdaGrad更新：参数 -= 学习率 / √(历史梯度平方和) * 当前梯度
        x1 -= eta / math.sqrt(s1 + eps) * g1  # 自适应调整x1的学习率
        x2 -= eta / math.sqrt(s2 + eps) * g2  # 自适应调整x2的学习率
        return x1, x2, s1, s2  # 返回更新后的参数和状态

    def f_2d(x1, x2):
        '''目标函数：椭圆抛物面'''
        return 0.1 * x1 ** 2 + 2 * x2 ** 2  # 最小值在(0,0)

    eta = 0.4 # 中等学习率
    common.show_trace_2d(f_2d, common.train_2d(adagrad_2d))

    eta = 2 # 大学习率（测试AdaGrad的鲁棒性）
    common.show_trace_2d(f_2d, common.train_2d(adagrad_2d))


    def init_adagrad_states(feature_dim):
        '''初始化AdaGrad的状态变量（梯度平方累积器）
        为每个可训练参数创建对应的状态变量，用于记录历史梯度平方和
        feature_dim: 特征维度（输入特征数量）
        返回：(s_w, s_b): 权重和偏置的梯度平方累积状态
        '''
        s_w = torch.zeros((feature_dim, 1))  # 权重的梯度平方累积器（与w同形状）
        s_b = torch.zeros(1)                 # 偏置的梯度平方累积器（与b同形状）
        return (s_w, s_b)

    def adagrad(params, states, hyperparams):
        '''AdaGrad优化器实现
        params: 待优化参数列表 [w, b]
        states: 状态变量列表 [s_w, s_b]
        hyperparams: 超参数字典 {'lr': 学习率}
        '''
        eps = 1e-6  # 小常数，防止除以零
        for p, s in zip(params, states):  # 同时遍历参数和对应的状态
            with torch.no_grad():         # 禁用梯度跟踪（纯数值计算）
                # 累积梯度平方：s = s + (梯度)^2
                s[:] += torch.square(p.grad)  # 原地更新，保持内存引用

                # AdaGrad更新：参数 -= 学习率 * 梯度 / √(s + ε)
                p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
            p.grad.data.zero_() # 清零当前梯度，准备下一轮计算

    # 获取数据迭代器和特征维度（批量大小=10）
    data_iter, feature_dim = common.get_data_ch11(downloader, batch_size=10)
    common.train_ch11(adagrad, init_adagrad_states(feature_dim),
                   {'lr': 0.1}, data_iter, feature_dim)

    # 使用PyTorch官方实现的AdaGrad（对比验证）
    trainer = torch.optim.Adagrad  # PyTorch内置AdaGrad优化器类
    common.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
# learn_AdaGrad_algorithm()









plt.tight_layout() # 自动调整子图参数，以避免标签、标题等元素重叠或溢出
plt.show()


plt.show(block=True)  # 阻塞显示，直到手动关闭窗口
# plt.pause(4444)  # 间隔的秒数： 4s
