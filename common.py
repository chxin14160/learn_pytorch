import time
import numpy as np

class Timer:  # @save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()



# import matplotlib.pyplot as plt
# import numpy as np
#
# axis_x = np.array([-8, -7, -6, -5, -4, -3, -2, -1])
# axis_y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# fig1 = plt.figure(1)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.plot(axis_x, axis_y)
# plt.draw()
# plt.pause(4)  # 间隔的秒数： 4s
# plt.close(fig1)
#
# fig2 = plt.figure(2)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.plot(axis_y, axis_x)
# plt.draw()
# plt.pause(6)  # 间隔的秒数：6s
# plt.close(fig1)

