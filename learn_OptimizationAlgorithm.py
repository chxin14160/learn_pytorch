import numpy as np
import torch
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import common



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







plt.tight_layout() # 自动调整子图参数，以避免标签、标题等元素重叠或溢出
plt.show()


plt.pause(4444)  # 间隔的秒数： 4s
