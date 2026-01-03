"""
脑肿瘤MRI影像数据集加载器
支持自动加载Training和Testing目录下的4类脑肿瘤影像
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BrainTumorDataset(Dataset):
    """
    脑肿瘤MRI影像分类数据集
    
    数据结构:
        data_root/
            Training/
                glioma_tumor/
                meningioma_tumor/
                no_tumor/
                pituitary_tumor/
            Testing/
                glioma_tumor/
                meningioma_tumor/
                no_tumor/
                pituitary_tumor/
    
    Args:
        data_root: 数据根目录路径
        split: 'train' 或 'test'
        transform: 图像转换操作
        target_transform: 标签转换操作
    """
    
    # 类别映射
    CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__()
        
        assert split in ['train', 'test'], f"split must be 'train' or 'test', got {split}"
        
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 确定数据目录
        if split == 'train':
            self.data_dir = self.data_root / 'Training'
        else:
            self.data_dir = self.data_root / 'Testing'
        
        # 加载所有图像路径和标签
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """扫描目录并加载所有样本"""
        for class_name in self.CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            class_idx = self.CLASS_TO_IDX[class_name]
            
            # 支持常见图像格式
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), class_idx))
        
        if len(self.samples) == 0:
            raise RuntimeError(f"未在 {self.data_dir} 找到任何图像文件")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本
        
        Returns:
            image: 转换后的图像张量
            label: 类别标签（整数）
        """
        img_path, label = self.samples[index]
        
        # 加载图像（转为RGB）
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_distribution(self):
        """获取数据集的类别分布"""
        from collections import Counter
        labels = [label for _, label in self.samples]
        return Counter(labels)


def get_transform(image_size: int = 224, split: str = 'train'):
    """
    获取数据转换pipeline
    
    Args:
        image_size: 目标图像大小
        split: 'train' 或 'test'，训练集会使用数据增强
    
    Returns:
        torchvision.transforms.Compose
    """
    if split == 'train':
        # 训练集：数据增强
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # 测试集：仅标准化
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # 测试数据集加载
    data_root = "/home/runner/work/NN_transformer/NN_transformer/data/raw"
    
    train_dataset = BrainTumorDataset(
        data_root=data_root,
        split='train',
        transform=get_transform(224, 'train')
    )
    
    test_dataset = BrainTumorDataset(
        data_root=data_root,
        split='test',
        transform=get_transform(224, 'test')
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"训练集类别分布: {train_dataset.get_class_distribution()}")
    print(f"测试集类别分布: {test_dataset.get_class_distribution()}")
    
    # 测试加载一个样本
    img, label = train_dataset[0]
    print(f"图像形状: {img.shape}, 标签: {label}")
