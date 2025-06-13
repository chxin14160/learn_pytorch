import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import common
from common import C_Downloader


''' 加载数据集 '''

downloader = C_Downloader()
DATA_HUB = downloader.DATA_HUB
DATA_URL = downloader.DATA_URL

# 下载并缓存Kaggle房屋数据集
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


# 调用download函数下载文件
cache_dir=os.path.join('.', 'data', 'kaggle_house') # 缓存路径为 .\data\kaggle_house
trainData_path = downloader.download('kaggle_house_train', cache_dir)
testData_path = downloader.download('kaggle_house_test', cache_dir)
print(f'【训练集】文件已下载到：{trainData_path}')
print(f'【测试集】文件已下载到：{testData_path}')

# 使用pandas分别加载包含训练数据和测试数据的两个CSV文件
train_data = pd.read_csv(trainData_path)
test_data = pd.read_csv(testData_path)
# train_data = pd.read_csv(downloader.download('kaggle_house_train'))
# test_data = pd.read_csv(downloader.download('kaggle_house_test'))

print(f"【训练集】：{train_data.shape} 包括1460个样本，每个样本80个特征和1个标")
print(f"【测试集】：{test_data.shape} 包含1459个样本，每个样本80个特征")
# 选择训练集的前4行样本数据，以及第[0, 1, 2, 3, -3, -2, -1]列的特征打印显示
print(f"查看前四个和最后两个特征，以及相应标签（房价）\n{train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]}")

# 对于每个样本：删除第一个特征ID，因为其不携带任何用于预测的信息
# .iloc 是 Pandas 提供的基于整数位置的索引器，用于按位置选择行和列
# train_data.iloc[:, 1:-1]，选择训练集的所有行，以及从第 2 列到倒数第 2 列的数据
# test_data.iloc[:, 1:]，选择了测试集的所有行，以及从第 2 列到最后一列的数据
# pd.concat：Pandas 提供的函数，用于沿指定轴（默认是 axis=0，即行方向）合并多个 DataFrame 或 Series
# 将删除ID(训练集还删除了标签)后的两个数据集合并在一起
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(f"删除无用标签后，前四个和最后两个特征，以及相应标签\n{all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]}")



''' 数据预处理 '''
# 若无法获得测试数据，则可根据训练数据计算均值和标准差：x←(x-μ)/σ
# 获取无法获得测试数据的数量

# 选择all_features 中所有数值类型的列（排除非数值列，如字符串或对象类型）
# all_features.dtypes：获取 all_features 中每一列的数据类型
# all_features.dtypes != 'object'：筛选出非对象类型（即数值类型）的列
# .index：获取这些数值列的列名
# numeric_features中的内容是：数字类型列的名字
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index    # 选择数值特征
# print(f"数值列下标：\n{numeric_features}")

# 对数值特征进行 Z-score 标准化（均值为 0，标准差为 1）
# apply(lambda x: ...)：对每一列应用标准化公式
# (x - x.mean()) / (x.std())：标准化公式[(原值-均值)/标准差]，其中：
# x.mean() 是列的均值
# x.std() 是列的标准差
# 标准化后，数据的均值为 0，标准差为 1
all_features[numeric_features] = all_features[numeric_features].apply(          # 标准化数值特征
    lambda x: (x - x.mean()) / (x.std()))
print(f"数据标准化后：\n{all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]}")

# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0（将标准化后的数值特征中的缺失值填充为 0）
# fillna(0)：将缺失值（NaN）替换为 0
# 标准化后均值为 0，因此填充 0 是合理的（不会引入偏差）
all_features[numeric_features] = all_features[numeric_features].fillna(0)       # 处理缺失值
print(f"处理缺失值后：\n{all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]}")

# 将分类特征（包括缺失值）转换为独热编码形式
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
# all_features是删除ID那一列之后，将每个样本中所有的特征连接起来
# pd.get_dummies()：Pandas 提供的函数，用于将分类变量转换为虚拟变量（独热编码）
# dummy_na=True：将缺失值（NaN）视为一个有效的类别，并为其创建一个额外的列（如 columnname_nan）
# 例如：某列有(A、B、缺失值)三种分类值 ，独热编码后会生成三列：columnname_A、columnname_B 和 columnname_nan
all_features = pd.get_dummies(all_features, dummy_na=True)                      # 独热编码分类特征
print(f"独热编码分类特征后：{all_features.shape}\n{all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]}")


# 通过values属性将数据从pandas格式提取numpy格式，并将其转为张量用于训练
n_train = train_data.shape[0] # 获取训练集的行数，即训练集的样本总数
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

# 获取训练集的价格标签，将其转换为张量
# .reshape(-1, 1)：-1表示自动计算该维度的大小(这里是样本数 n_train)，1表示只有一列(加个标签)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


"""
训练一个带有损失平方的线性模型：
1、损失函数为损失平方
2、线性模型作为基线模型
"""
loss = nn.MSELoss() # 使用均方误差损失函数(适用于回归任务)
in_features = train_features.shape[1] # 输入特征的维度，即为特征总数

def get_net(): # 仅返回一个简单的线性模型，将输入映射到1给输出值
    net = nn.Sequential(nn.Linear(in_features,1))
    return net


# 采用价格预测的对数来衡量差异：√￣(1/n*(∑(logy - logy')^2))
def log_rmse(net, features, labels): # 计算对数均方根误差（Log-RMSE）
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    # torch.clamp：将预测值裁剪到 [1, +∞) 范围内，避免对 0 或负数取对数（数学上无定义）
    clipped_preds = torch.clamp(net(features), 1, float('inf')) # 对数裁剪和变换
    # torch.log：对裁剪后的预测值和真实标签取自然对数
    # torch.sqrt：对 MSE 取平方根，得到 RMSE
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                          torch.log(labels)))
    return rmse.item() # 将张量转换为 Python 标量值（浮点数）


# 优化器借助Adam优化器
"""
定义训练函数：
1、加载训练数据集
2、使用Adam优化器（对初始学习率不那么敏感）
3、进行训练：计算损失，进行梯度优化，返回训练损失和测试损失
"""
def train(net,
          train_features, train_labels,
          test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # TensorDataset：将 train_features 和 train_labels 包装为 PyTorch 的数据集
    # DataLoader：按 batch_size 分批加载数据，并打乱顺序（shuffle=True）
    train_iter = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size, shuffle=True)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay) # 权重衰减（L2 正则化系数）
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()   # 梯度归零(清除梯度)
            l = loss(net(X), y)     # 计算损失
            l.backward()            # 更新梯度(反向传播计算梯度)
            optimizer.step()        # 更新参数

        # 将训练损失加到训练损失列表中
        train_ls.append(log_rmse(net, train_features, train_labels))

        # 如果测试标签不为空，将测试损失加到测试损失列表中
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls # 返回训练集和测试集的损失列表


"""
定义K折交叉验证:
1、当k > 1时，进行K折交叉验证，将数据集分为K份
2、选择第i个切片作为验证集，其余部分作为训练数据
3、第一片的训练数据直接填进去，之后的使用cat进行相连接
"""
def get_k_fold_data(k, i, X, y):
    # 假设k折大于1，将数据集分成k份
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]

        # 将第i片数据集作为验证集
        if j == i:
            X_valid, y_valid = X_part, y_part

        # 其他余下的片作为训练集
        elif X_train is None:
            X_train, y_train = X_part, y_part

        # 使用cat()函数将余下的训练集片进行拼接
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid











