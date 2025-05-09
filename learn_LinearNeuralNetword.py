from common import Timer
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

''' 矢量化加速例子 开始 '''
# # 实例化两个全为1的10000维向量
# n = 10000
# a = torch.ones([n])
# b = torch.ones([n])
#
# # 开始对工作负载进行基准测试
# c = torch.zeros(n)
# timer = Timer()
#
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f"方法一：使用循环遍历向量，耗时：{timer.stop():.5f} sec")
#
# timer.start()
# d = a + b
# print(f"方法二：使用重载的+运算符来计算按元素的和，耗时：{timer.stop():.5f} sec")
''' 矢量化加速例子 结束 '''


''' 正态分布与平方缺失 开始 '''
# # 计算正态分布
# def normal(x, mu, sigma):
#     p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
#     return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
#
# # 可视化不同参数的正态分布（高斯分布）概率密度函数
# x = np.arange(-7, 7, 0.01) # 生成数据点(即 创建从-7到7（步长0.01）的等间距数组，共1400个点，作为横坐标
# params = [(0, 1), (0, 2), (3, 1)] # 三个正态分布的参数对(均值mu，标准差sigma)
# # 第一个：μ=0, σ=1（标准正态分布）
# # 第二个：μ=0, σ=2（更宽的分布）
# # 第三个：μ=3, σ=1（向右平移的分布）
#
# # 可视化
# plt.figure(figsize=(4.5, 2.5))
# for mu,sigma in params:
#     plt.plot(x, normal(x, mu, sigma), label=f'mean{mu}, std{sigma}')
#
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.xlabel('x轴')
# plt.ylabel('p(x)')
# plt.legend()
# plt.grid() # 添加网格
# plt.show()
''' 正态分布与平方缺失 结束 '''



