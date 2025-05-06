# %matplotlib inline
import torch

fair_probs = torch.ones([6]) / 6 # 每个类别的概率分布，即 均匀分布的概率张量 [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
print(f"每个类别的概率分布: {fair_probs}  \nfair_probs形状: {fair_probs.shape}")
sample = torch.multinomial(fair_probs, 1)  # 注意：直接使用 torch.multinomial 更常见
print(f"\n从多项分布中抽取一个样本: {sample}, 表示抽到了骰子的第{sample.item()}面  \nsample形状: {sample.shape}")

sample = torch.multinomial(fair_probs, 10, replacement=True)  # , replacement=True表示允许重复采样
print(f"\n每次试验抽取10个样本(允许重复采样): {sample},10次抽取中每次被抽到的面 \nsample形状: {sample.shape}")

# 将结果存储为32位浮点数以进行除法
counts = torch.multinomial(fair_probs, 2000, replacement=True) # 从多项分布中采样1000次
record = torch.zeros([6])
for cur in counts:
    record[cur] += 1 # 统计各个面分别出现的次数
relative_frequencies = record / 2000  # 计算相对频率作为估计值
print("从多项分布中采样2000次，"
      "其中每个元素表示一次采样的结果（0到5之间的整数，对应骰子的六个面）")
print("Counts:", counts)
print("record:", record)
print("Relative frequencies:", relative_frequencies)

