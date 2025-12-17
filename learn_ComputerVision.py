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
plt.show()

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


def learn_Multi_GPU_training():
    '''图像增广：多GPU训练部分'''
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


    # 超参数设置
    batch_size = 256    # 总batch大小（会被自动分割到多个GPU）
    # batch_size = 32     # 我GPU只有2GB显存，所以需要将批次大小减小
    devices = common.try_all_gpus() # 自动检测可用GPU
    net = common.resnet18(10, 3)    # 10分类，3通道输入（CIFAR-10）

    # Xavier均匀初始化（适合tanh/sigmoid激活函数）
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(m.weight) # 保持输入输出方差一致

    net.apply(init_weights) # 递归应用初始化函数到所有子模块

    # 数据增强训练封装函数
    def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
        """支持数据增强的完整训练流程"""
        # 数据加载（应用数据增强）
        # train_iter = common.load_cifar10(True, train_augs, batch_size, learn_pytorch=True)   # 训练集
        # test_iter = common.load_cifar10(False, test_augs, batch_size, learn_pytorch=True)    # 测试集
        train_iter = common.load_cifar10(True, train_augs, batch_size)   # 训练集
        test_iter = common.load_cifar10(False, test_augs, batch_size)    # 测试集
        # 损失函数和优化器
        loss = nn.CrossEntropyLoss(reduction="none")        # 不自动求平均
        trainer = torch.optim.Adam(net.parameters(), lr=lr) # Adam优化器
        # 启动训练
        # epoch_train_losses, epoch_train_accs, epoch_test_accs = (
        #     common.train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices))
        common.train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

    train_with_data_aug(train_augs, test_augs, net)
# learn_Multi_GPU_training()







# 注册数据集信息到DATA_HUB全局字典
# 格式：(数据集URL, MD5校验值)
DATA_HUB['hotdog'] = (DATA_URL + 'hotdog.zip', # 完整下载URL（DATA_URL是d2l定义的基准URL）
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5') # 文件MD5，用于校验下载完整性

# 下载并解压数据集（若本地不存在）：自动从DATA_HUB下载压缩包并解压到本地缓存目录
data_dir = downloader.download_extract('hotdog')

def learn_Hot_dog_recognition():
    '''演示微调：热狗识别'''
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

    # 加载预训练模型：下载并加载在ImageNet上预训练的ResNet-18模型
    ''' pretrained=True 使用在ImageNet上训练好的权重
    加载过程：
    1. 检查本地缓存是否有预训练权重
    2. 若没有，自动从PyTorch服务器下载
    3. 加载权重到模型结构
    4. 返回完整的预训练模型
    '''
    pretrained_net = torchvision.models.resnet18(pretrained=True)

    print(f"模型的最后一层(全连接层)结构：{pretrained_net.fc}")

    # 创建微调模型(重新加载模型，而非在原有模型上修改)
    finetune_net = torchvision.models.resnet18(pretrained=True)
    # 修改分类层：将1000类分类器替换为2类分类器（热狗 vs 非热狗）
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    # 初始化新分类层：使用Xavier均匀分布初始化新分类层的权重
    nn.init.xavier_uniform_(finetune_net.fc.weight)


    # 若param_group=True，则输出层中的模型参数将使用十倍的学习率（差异化学习率）
    def train_fine_tuning(net, learning_rate,           # 要训练的模型(已修改最后一层)，学习率
                          batch_size=128, num_epochs=5, # 批次大小，训练轮数
                          param_group=True):            # 是否使用参数分组（差异化学习率）
        # 创建数据加载器
        train_iter = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                os.path.join(data_dir, 'train'),
                transform=train_augs), # 训练集增强
            batch_size=batch_size,
            shuffle=True) # 训练集打乱
        test_iter = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                os.path.join(data_dir, 'test'),
                transform=test_augs), # 测试集增强
            batch_size=batch_size,
            shuffle=False) # 测试集不打乱
        devices = common.try_all_gpus() # 设备配置：自动检测可用的GPU设备，支持多GPU训练
        # 默认：reduction="mean"→ 返回批次损失的平均值
        # 当前：reduction="none"→ 返回每个样本的损失
        loss = nn.CrossEntropyLoss(reduction="none")
        if param_group: # 若需参数分组
            # 提取特征提取器参数（除最后一层外的所有参数）
            params_1x = [param for name, param in net.named_parameters()
                 if name not in ["fc.weight", "fc.bias"]]
            # 创建优化器，不同参数组使用不同学习率
            trainer = torch.optim.SGD([
                # 组1：特征提取器参数（小学习率微调）(初始化为源模型相应层的模型参数 ∴只需微调)
                {'params': params_1x}, # 使用外层lr（默认），即learning_rate小学习率
                # 组2：分类器参数（大学习率快速学习）(新的输出层参数随机初始化 ∴需大步进以加快速度)
                {'params': net.fc.parameters(), 'lr': learning_rate * 10}
            ], lr=learning_rate, weight_decay=0.001)
        else: #  统一学习率（对比基准）
            trainer = torch.optim.SGD(net.parameters(),     # 所有参数
                                      lr=learning_rate,     # 统一学习率
                                      weight_decay=0.001)   # L2正则化
        common.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                       devices)

    # train_fine_tuning(finetune_net, 5e-5) # 开始训练
# learn_Hot_dog_recognition()



def learn_object_detection_and_bounding_boxes():
    '''目标检测和边界框'''
    plt.figure(figsize=(5, 3))
    img = plt.imread('./img/catdog.jpg') # 当前py文件同路径的img文件夹中
    plt.imshow(img)

    # bbox是边界框的英文缩写
    # dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
    dog_bbox, cat_bbox = [50.0, 35.0, 420.0, 550.0], [435.0, 115.0, 723.0, 520.0]

    # 将边界框列表转换为PyTorch张量
    boxes = torch.tensor((dog_bbox, cat_bbox))
    # 验证转换的可逆性：角点↔中心点↔角点应该得到原始值
    same = common.box_center_to_corner(common.box_corner_to_center(boxes)) == boxes
    print(f"角点↔中心点↔角点后是否得到原始值：\n{same}")

    # 显示原始图像，并在图上添加狗(蓝)猫(红)的边界框
    fig = plt.imshow(img)
    fig.axes.add_patch(common.bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(common.bbox_to_rect(cat_bbox, 'red'))
# learn_object_detection_and_bounding_boxes()



img = plt.imread('./img/catdog.jpg')
# plt.figure(figsize=(5, 3)) # 创建新画布
plt.imshow(img)
plt.title('catdog')
# plt.axis('off')  # 可选：隐藏坐标轴
# plt.show()
h, w = img.shape[:2]

print(f"原图尺寸： 高{h}, 宽{w}")
# 生成 [0,1) 均匀分布的随机数，创建随机图像张量作为输入(batch_size, channels, height, width)
X = torch.rand(size=(1, 3, h, w)) # 批次大小为1，RGB3通道
Y = common.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(f"生成的锚框变量形状：{Y.shape}")

# 5 = boxes_per_pixel = len(sizes) + len(ratios) - 1 = 3+3-1
boxes = Y.reshape(h, w, 5, 4) # 将形状改为(图像高度, 图像宽度, 以同一像素为中心的锚框的数量, 4)
print(f"以（250,250）为中心的第一个锚框：{boxes[250, 250, 0, :]}")

plt.figure(figsize=(5, 3)) # 创建新画布
bbox_scale = torch.tensor((w, h, w, h)) # 准备坐标缩放，以便将归一化坐标转换为像素坐标
fig = plt.imshow(img)
# boxes[250, 250, :, :] 是一个二维张量torch.Size([5, 4])
common.show_bboxes(fig.axes,  # 图像所在的坐标轴对象
                   boxes[250, 250, :, :] * bbox_scale, # 提取像素位置(250,250)处的所有5个锚框后坐标转换，归一化→像素坐标
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])









plt.tight_layout() # 自动调整子图参数，以避免标签、标题等元素重叠或溢出
plt.show()

plt.show(block=True)  # 阻塞显示，直到手动关闭窗口
