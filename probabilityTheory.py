# %matplotlib inline
import torch
import matplotlib.pyplot as plt

fair_probs = torch.ones([6]) / 6 # 每个类别的概率分布，即 均匀分布的概率张量 [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
print(f"每个类别的概率分布: {fair_probs}  \nfair_probs形状: {fair_probs.shape}")
sample = torch.multinomial(fair_probs, 1)  # 注意：直接使用 torch.multinomial 更常见
print(f"\n从多项分布中抽取一个样本: {sample}, 表示抽到了骰子的第{sample.item()}面  \nsample形状: {sample.shape}")

sample = torch.multinomial(fair_probs, 10, replacement=True)  # , replacement=True表示允许重复采样
print(f"\n每次试验抽取10个样本(允许重复采样): {sample},10次抽取中每次被抽到的面 \nsample形状: {sample.shape}")


# 将结果存储为32位浮点数以进行除法
counts = torch.multinomial(fair_probs, 2000, replacement=True) # 从多项分布中采样2000次
record = torch.zeros([6])
for cur in counts:
    record[cur] += 1 # 统计各个面分别出现的次数
relative_frequencies = record / 2000  # 计算相对频率作为估计值
print("从多项分布中采样2000次，"
      "其中每个元素表示一次采样的结果（0到5之间的整数，对应骰子的六个面）")
print("Counts:", counts)
print("record:", record)
print("Relative frequencies:", relative_frequencies)


# 设置随机种子以便复现结果
# torch.manual_seed(42)

# 从多项分布中采样500次，每次采样cnt次(允许重复采样)
cnt = 10 # 每次试验的采样次数

# .repeat(500, 1) 将采样结果重复500次，得到一个形状为 [500, cnt] 的张量
# 实现发现，500次中每次的采样结果皆相同
# counts = torch.multinomial(fair_probs, 10, replacement=True).repeat(500, 1)
# 使用列表推导式，500次中每次采样结果皆不同(随机)
counts = torch.stack([torch.multinomial(fair_probs, cnt, replacement=True) for _ in range(500)])
print(f"counts:{counts}\n"
      f"counts形状为{counts.shape} ，其中每一行表示一次试验中 {cnt}次采样的结果") # 元素即为抽中的面

cum_counts = torch.zeros([500,6])
for col in range(500):
    for cur in counts[col]:
        cum_counts[col][cur] += 1 # 统计500次抽样中，每次各个面的抽中次数
print(f"\n500次抽样中，每次各个面的抽中次数 cum_counts({cum_counts.shape}): \n{cum_counts}")

cum_counts = cum_counts.cumsum(dim=0) # 计算累积计数，.cumsum(dim=0) 沿着第0维（行）计算累积和(保持维度不变，计算累计到当前次的各个面抽中次数)
print(f"\n累计到当前次的各个面抽中次数 cum_counts({cum_counts.shape}): \n{cum_counts}")

# for col in range(500):
#     cum_counts[col] /= cnt # 计算500次中每次 各个面的相对频率值作为估计值
# print(f"500次中每次，累计到当前次的各个面的相对频率值(作为估计值)({cum_counts.shape}): \n{cum_counts}")

# -cum_counts.sum(dim=1, keepdims=True) 计算每一行（累计到当前次试验）的采样总数，并保持维度不变
# -cum_counts / cum_counts.sum(dim=1, keepdims=True) 计算每个面的估计概率
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True) # 计算估计概率
print(f"\n每个元素皆为【累计到当前试验，某个面的总出现概率】estimates({estimates.shape}):\n{estimates}")

# 设置图形大小
plt.figure(figsize=(6, 4.5))

# 绘制每个面的估计概率
# 总共6列6个面6条线，转换后每列变成：x是行数即第几次试验，y是累计到当前试验的面出现概率
for i in range(6):
    plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
    # plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
    # estimates[:, i] 选取估计概率张量中第 i 列，表示第 i+1 个面的估计概率
    # .numpy() 将张量转换为NumPy数组，以便使用Matplotlib进行绘图

# 绘制理论概率的虚线
plt.axhline(y=1/6, color='black', linestyle='dashed') # 在y=1/6处绘制，黑色，虚线
plt.xlabel('Groups of experiments') # x轴标签：试验组
plt.ylabel('Estimated probability') # y轴标签：估计概率
plt.legend()    # 为图表添加图例
plt.show()      # 显示图表（显示在屏幕上）


