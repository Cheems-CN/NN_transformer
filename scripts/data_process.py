from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 填入你解压后的 Training 文件夹路径
train_dataset = datasets.ImageFolder(root=r'..\data\raw\Training')
val_dataset = datasets.ImageFolder(root=r'..\data\raw\Training')

print(f"找到的所有类别: {train_dataset.classes}")
print(f"类别与数字的对应关系: {train_dataset.class_to_idx}")
print(f"第一张图片的路径和标签索引: {train_dataset.imgs[0]}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 增强
    transforms.RandomRotation(10),     # 增强
    transforms.ToTensor(),             # 必选
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 必选
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_loader = DataLoader(
    dataset=train_dataset,      # 刚才建好的仓库
    batch_size=32,             # 每次搬多少货
    shuffle=True,              # 是否打乱顺序
    num_workers=0,             # 找几个“搬运工”并行读数据
    pin_memory=True            # 锁页内存（能让数据从内存进显存更快）
)

val_loader = DataLoader(
    dataset=val_dataset,      # 刚才建好的仓库
    batch_size=32,             # 每次搬多少货
    shuffle=True,              # 是否打乱顺序
    num_workers=0,             # 找几个“搬运工”并行读数据
    pin_memory=True            # 锁页内存（能让数据从内存进显存更快）
)


train_dataset.transform = train_transform
val_dataset.transform = val_transform

# --- 1. 获取一个 Batch 的数据 ---
data_iter = iter(train_loader)
images, labels = next(data_iter)


# --- 2. 定义反归一化函数 ---
def imshow(img, title=None):
    # 将 Tensor [C, H, W] 转回 [H, W, C]
    img = img.numpy().transpose((1, 2, 0))

    # 反归一化：img = std * normalized_img + mean
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean

    # 修正像素范围在 0-1 之间，防止警告
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # 关掉坐标轴


# --- 3. 画出一组图 (展示前 8 张) ---
plt.figure(figsize=(15, 8))
class_names = train_dataset.classes  # 获取 ['glioma_tumor', ...]

for i in range(8):
    plt.subplot(2, 4, i + 1)
    imshow(images[i], title=class_names[labels[i]])

plt.tight_layout()
plt.show()

print("可视化检查完成！如果图片清晰且标签对应正确，说明 Data Pipeline 已通。")