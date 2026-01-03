"""
统一训练框架
支持训练三种模型：PureCNN、PureViT、HybridModel
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.brain_tumor_dataset import BrainTumorDataset, get_transform
from src.models.pure_cnn import PureCNN
from src.models.vit import VisionTransformer
from src.models.hybrid_model import HybridModel
from src.models.parallel_model import ParallelModel
from src.models.embedded_model import EmbeddedModel


class Trainer:
    """统一训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir,
        model_name
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"开始训练模型: {self.model_name}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 打印摘要
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✅ 新的最佳模型! Val Acc: {val_acc:.2f}%")
            
            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # 保存训练历史和曲线
        self.save_history()
        self.plot_curves()
        
        print(f"\n{'='*60}")
        print(f"训练完成! 最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"模型保存在: {self.save_dir}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            path = self.save_dir / f'{self.model_name}_best.pth'
        else:
            path = self.save_dir / f'{self.model_name}_epoch{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.save_dir / f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def plot_curves(self):
        """绘制训练曲线"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss曲线
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.model_name} - Loss Curves')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy曲线
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{self.model_name} - Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{self.model_name}_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_model(model_type, num_classes=4):
    """创建模型"""
    if model_type == 'pure_cnn':
        model = PureCNN(num_classes=num_classes)
    elif model_type == 'pure_vit':
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            n_classes=num_classes,
            d_model=512,
            n_layers=8,
            n_head=8,
            d_ff=2048,
            dropout=0.1
        )
    elif model_type == 'hybrid':
        model = HybridModel(
            num_classes=num_classes,
            d_model=512,
            n_layers=4,
            n_head=8,
            d_ff=2048,
            dropout=0.1
        )
    elif model_type == 'parallel':
        model = ParallelModel(
            num_classes=num_classes,
            feature_dim=512,
            vit_d_model=512,
            vit_n_layers=6,
            vit_n_head=8,
            vit_d_ff=2048,
            dropout=0.1
        )
    elif model_type == 'embedded':
        model = EmbeddedModel(
            num_classes=num_classes,
            base_channels=32,
            transformer_channels=256,
            n_head=4,
            n_transformer_layers=2,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='训练脑肿瘤分类模型')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['pure_cnn', 'pure_vit', 'hybrid', 'parallel', 'embedded'],
                       help='模型类型')
    parser.add_argument('--data_root', type=str, 
                       default='/home/runner/work/NN_transformer/NN_transformer/data/raw',
                       help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("加载数据集...")
    train_dataset = BrainTumorDataset(
        data_root=args.data_root,
        split='train',
        transform=get_transform(224, 'train')
    )
    
    val_dataset = BrainTumorDataset(
        data_root=args.data_root,
        split='test',
        transform=get_transform(224, 'test')
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"训练集类别分布: {train_dataset.get_class_distribution()}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = create_model(args.model, num_classes=4)
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        model_name=args.model
    )
    
    # 开始训练
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
