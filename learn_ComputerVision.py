import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import common

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(5, 3))
# plt.rcParams['figure.figsize'] = (5, 3)
img = Image.open('./img/cat1.jpg') # 当前py文件同路径的img文件夹中
plt.imshow(img)
# plt.axis('off')  # 隐藏坐标轴
plt.title('Original Image')
plt.show()

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5,
          title=None, titles=None):
    """应用数据增强并显示结果"""
    Y = [aug(img) for _ in range(num_rows * num_cols)] # 生成增强后的图像
    common.show_images(Y, num_rows, num_cols, scale=scale,
                       title=title, titles=titles) # 显示结果

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








