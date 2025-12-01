import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import common
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 下载器与数据集配置
# 为数据集注册下载信息，包括文件路径和校验哈希值（用于验证文件完整性）
downloader = common.C_Downloader()
DATA_HUB = downloader.DATA_HUB  # 字典，存储数据集名称与下载信息
DATA_URL = downloader.DATA_URL  # 基础URL，指向数据集的存储位置



plt.figure(figsize=(5, 3))
# plt.rcParams['figure.figsize'] = (5, 3)
img = Image.open('./img/cat1.jpg') # 当前py文件同路径的img文件夹中
plt.imshow(img)
# plt.axis('off')  # 隐藏坐标轴
plt.title('Original Image')
# plt.show()

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5,
          title=None, titles=None):
    """应用数据增强并显示结果"""
    Y = [aug(img) for _ in range(num_rows * num_cols)] # 生成增强后的图像
    common.show_images(Y, num_rows, num_cols, scale=scale,
                       title=title, titles=titles) # 显示结果

def demonstrate_augmentation_methods():
    '''演示：图像增广方法'''
    # 水平翻转增强(左右翻转)
    print("=== 随机水平翻转 ===")
    horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5) # 翻转概率0.5
    apply(img, horizontal_flip, title='概率0.5，随机水平翻转')

    # 垂直翻转增强(上下翻转)
    print("=== 随机垂直翻转 ===")
    vertical_flip = torchvision.transforms.RandomVerticalFlip(p=0.5) # 翻转概率0.5
    apply(img, vertical_flip, title='概率0.5，随机垂直翻转')

    # 随机裁剪和缩放(随机裁剪并缩放到指定尺寸)
    # 尺度不变性：学习不同大小的物体
    # 位置不变性：物体在不同位置都能识别
    print("=== 随机尺寸裁剪 ===")
    shape_aug = torchvision.transforms.RandomResizedCrop(
            size=(200, 200),    # 输出尺寸
            scale=(0.1, 1),     # 裁剪面积比例范围：10%-100%
            ratio=(0.5, 2))     # 宽高比范围：0.5:1 到 2:1
    apply(img, shape_aug, title='随机裁剪并缩放到指定尺寸')

    # 亮度调整(模拟不同光照条件)
    print("=== 亮度调整 (brightness=0.5) ===")
    brightness_aug = torchvision.transforms.ColorJitter(
        brightness=0.5, # 亮度变化幅度 (±0.5)
        contrast=0,     # 对比度不变
        saturation=0,   # 饱和度不变
        hue=0)          # 色调不变
    apply(img, brightness_aug, title='亮度调整 (brightness=0.5)')

    # 色调调整
    print("=== 色调调整 (hue=0.5) ===")
    hue_aug = torchvision.transforms.ColorJitter(
        brightness=0,   # 亮度不变
        contrast=0,     # 对比度不变
        saturation=0,   # 饱和度不变
        hue=0.5)        # 色调变化幅度 (±0.5)
    apply(img, hue_aug, title='色调调整 (hue=0.5)')

    # 综合颜色调整 (变化幅度皆为±0.5)
    print("=== 综合颜色调整 ===")
    color_aug = torchvision.transforms.ColorJitter(
        brightness=0.5,  # 亮度
        contrast=0.5,    # 对比度
        saturation=0.5,  # 饱和度
        hue=0.5)         # 色调
    apply(img, color_aug, title='综合颜色调整')

    # 组合增强（数据增强流水线）：一次性应用多种增强，创建更丰富的训练数据
    print("=== 组合增强: 水平翻转 + 颜色调整 + 随机裁剪 ===")
    combined_augs = torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
        color_aug,      # 颜色调整
        shape_aug])     # 随机裁剪(形状增强)
    apply(img, combined_augs, title='组合增强，一次性应用多种增强')
# demonstrate_augmentation_methods()


all_images = torchvision.datasets.CIFAR10(
    train=True,           # 加载训练集（False为测试集）
    root="../data",       # 数据存储目录
    download=True)        # 若数据不存在，自动下载
common.show_images([all_images[i][0] for i in range(32)], # 获取前32张图像
                   4, 8,      # 4行8列网格
                   scale=0.8) # 图像缩放比例

# 训练集增强：只对训练样本图像增广，这里只使用最简单的随机左右翻转
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(), # 水平翻转（50%概率）
     torchvision.transforms.ToTensor()])            # 转为Tensor（0-1范围）

# 测试集增强：预测过程中不使用随机操作的图像增广，ToTensor将图像转为深度学习框架所要求的格式
test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()]) # 仅转换




# 注册数据集信息到DATA_HUB全局字典
# 格式：(数据集URL, MD5校验值)
DATA_HUB['hotdog'] = (DATA_URL + 'hotdog.zip', # 完整下载URL（DATA_URL是d2l定义的基准URL）
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5') # 文件MD5，用于校验下载完整性

# 下载并解压数据集（若本地不存在）：自动从DATA_HUB下载压缩包并解压到本地缓存目录
data_dir = downloader.download_extract('hotdog')

# 演示微调：热狗识别
# 使用ImageFolder加载数据集
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs  = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 准备可视化样本 并 显示对比图像
hotdogs     = [train_imgs[i][0] for i in range(8)]      # 前8张热狗图片
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)] # 后8张非热狗图片
common.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)


# 使用RGB通道的均值和标准差，以标准化每个通道
# 将图像从 [0,1] 范围标准化到接近 [-1,1]
# 标准化公式: (输入 - 均值) / 标准差
normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], # ImageNet数据集的均值
    std=[0.229, 0.224, 0.225])  # ImageNet数据集的标准差

# 训练数据处理：随机性 + 多样性 → 提高泛化
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),  # 随机缩放裁剪：随机裁剪大小长宽后缩放至目标尺寸(尺度不变性+位置不变性)
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转(镜像对称不变性)
    torchvision.transforms.ToTensor(),              # 转为Tensor(0-1范围float32),(C,H,W) [通道优先]
    normalize])                                     # 标准化(加速收敛 + 训练稳定)

# 测试数据处理：确定性 + 一致性 → 准确评估
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),  # 调整尺寸：统一缩放至224*224
    torchvision.transforms.CenterCrop(224),     # 中心裁剪：从图像中心裁剪固定区域
    torchvision.transforms.ToTensor(),          # 转为Tensor（0-1范围float32),(C,H,W) [通道优先]
    normalize])                                 # 标准化














