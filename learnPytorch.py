import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",  # 数据集存储的位置
    train=True,  # 加载训练集（True则加载训练集）
    download=True,  # 如果数据集在指定目录中不存在，则下载（True才会下载）
    transform=ToTensor(),  # 应用于图像的转换列表，例如转换为张量和归一化
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,  # 加载测试集（False则加载测试集）
    download=True,
    transform=ToTensor(),
)



batch_size = 64

# Create data loaders.
# DataLoader()：batch_size每个批次的大小，shuffle=True则打乱数据
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:  # 遍历训练数据加载器，x相当于图片，y相当于标签
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 使用加速器，并打印当前使用的加速器（当前加速器可用则使用当前的，否则使用cpu）
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # torch2.4.2并没有accelerator这个属性，2.6的才有，所以注释掉
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 检查 CUDA 是否可用
print("CUDA available:", torch.cuda.is_available())



# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)  # torch2.4.2并没有accelerator这个属性，2.6的才有，所以注释掉不用
# model = NeuralNetwork()
print(model)



loss_fn = nn.CrossEntropyLoss() # 损失函数，nn.CrossEntropyLoss()用于多分类
'''
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
优化器用PyTorch 提供的随机梯度下降（Stochastic Gradient Descent, SGD）优化器
model.parameters()：将模型的参数传递给优化器，优化器会根据这些参数计算梯度并更新它们
lr=1e-3：学习率（learning rate），控制每次参数更新的步长（较大的学习率可能导致训练不稳定，较小的学习率可能导致训练速度变慢）
'''
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # 优化器，用于更新模型的参数，以最小化损失函数


'''
 训练模型（单个epoch）
dataloader：数据加载器，用于按批次加载训练数据
model     ：神经网络模型
loss_fn   ：损失函数，用于计算预测值与真实值之间的误差
optimizer ：优化器，用于更新模型参数
'''
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y) # pred为模型的预测值即输出, y为实际的类别标签

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 测试模型
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 测试集的总样本数
    num_batches = len(dataloader)   # 测试数据加载器（dataloader）的总批次数
    model.eval()                    # 设置为评估模式，这会关闭 dropout 和 batch normalization 的训练行为
    test_loss, correct = 0, 0       # 累积测试损失和正确预测的样本数
    with torch.no_grad(): # 禁用梯度计算，使用 torch.no_grad() 上下文管理器，避免计算梯度，从而节省内存并加速计算
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # 将数据加载到指定设备
            pred = model(X) # 模型预测
            test_loss += loss_fn(pred, y).item() # 累积损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 累积正确预测数
            # correct += (pred.argmax(1) == y).float().sum().item()  # 可以直接使用 .float()，更简洁

    test_loss /= num_batches    # 平均损失
    correct /= size             # 准确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 跑5轮，每轮皆是先训练，然后测试
epochs = 7 # 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')