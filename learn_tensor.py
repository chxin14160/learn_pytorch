import torch
import numpy as np

# 直接从数据创建张量。数据类型是自动推断的
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# 从 NumPy 数组创建张量（反之亦然）
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


# 从另一个张量创建张量，新张量保留参数张量的属性（形状、数据类型），除非显式覆盖
x_ones = torch.ones_like(x_data) # retains the properties of x_data 保留原有属性
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data 覆盖原有类型
print(f"Random Tensor: \n {x_rand} \n")


# 使用随机值或常量值创建张量，shape是张量维度的元组。在下面的函数中，它确定输出张量的维数
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")               # 形状
print(f"Datatype of tensor: {tensor.dtype}")            # 数据类型
print(f"Device tensor is stored on: {tensor.device}")   # 存储其的设备


tensor = torch.ones(4, 4)
# tensor = torch.tensor([[1, 2, 3, 4],
#                        [5, 6, 7, 8],
#                        [9, 10, 11, 12],
#                        [13, 14, 15, 16]])
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


# 计算两个张量间的矩阵乘法
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` 返回张量的转置 returns the transpose of a tensor
y1 = tensor @ tensor.T          # “@”是矩阵乘法的简写，用于张量之间的矩阵乘法; tensor.T 返回 tensor 的转置
y2 = tensor.matmul(tensor.T)    # matmul用于矩阵乘法，与 @ 功能等价

y3 = torch.rand_like(y1)        # 创建与 y1 形状相同的新张量，元素为随机值
torch.matmul(tensor, tensor.T, out=y3)  # 进行矩阵乘法，并将结果存储在 y3 中

print("Matrix Multiplication Results:") #  y1, y2, y3 三者相等
print("y1:\n", y1)
print("y2:\n", y2)
print("y3:\n", y3)

# 计算元素积
# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor    # 对 tensor 进行逐元素相乘
z2 = tensor.mul(tensor) # 与 * 相同的逐元素相乘

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3) # 使用 torch.mul 函数对 tensor 进行逐元素相乘，并将结果存储在 z3 中

print("\nElement-wise Product Results:") #  z1, z2, z3 三者相等
print("z1:\n", z1)
print("z2:\n", z2)
print("z3:\n", z3)


agg = tensor.sum()      # 所有元素求和，返回新的张量（标量张量）
agg_item = agg.item()   # 将标量张量agg转成Python的基本数据类型（如float或int，具体取决于张量中数据的类型）
print(agg_item, type(agg_item))


# 使用 in-place 操作对张量中的每个元素加 5
print(f"{tensor} \n")
tensor.add_(5)  # add_ 是 in-place 操作，会直接修改原张量
print(tensor)



t = torch.ones(5)   # 创建一个包含 5 个 1.0 的张量
print(f"t: {t}")

# 将张量 t 转换为 NumPy 数组
n = t.numpy()       # .numpy() 方法将 PyTorch 张量转换为 NumPy 数组
print(f"n: {n}")

t.add_(1)           # 使用 add_ 进行原地加法
print(f"t: {t}")
print(f"n: {n}")    # n 的值也会改变，因为 t 和 n 共享内存



n = np.ones(5)          # 创建一个包含 5 个 1.0 的 NumPy 数组
t = torch.from_numpy(n) # 将 NumPy 数组转换为 PyTorch 张量
print(f"t: {t}")
print(f"n: {n}")

np.add(n, 5, out=n)     # 对 NumPy 数组，使用 out 参数 进行原地加法操作
print(f"t: {t}")        # 由于 t 和 n 底层共享内存，t 的值也会随之改变
print(f"n: {n}")