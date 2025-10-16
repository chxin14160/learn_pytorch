import numpy as np
import torch
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








plt.tight_layout() # 自动调整子图参数，以避免标签、标题等元素重叠或溢出
plt.show()


plt.pause(4444)  # 间隔的秒数： 4s
