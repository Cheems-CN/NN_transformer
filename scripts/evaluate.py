"""
模型评估脚本
计算准确率、精确率、召回率、F1分数，并生成混淆矩阵
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.brain_tumor_dataset import BrainTumorDataset, get_transform
from src.models.pure_cnn import PureCNN
from src.models.vit import VisionTransformer
from src.models.hybrid_model import HybridModel
from src.models.parallel_model import ParallelModel
from src.models.embedded_model import EmbeddedModel


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


def evaluate_model(model, test_loader, device, class_names):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("评估模型...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 详细的分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class': {
            class_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return results, cm


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵已保存到: {save_path}")


def print_results(results, model_name):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    print(f"总体准确率: {results['accuracy']*100:.2f}%")
    print(f"加权精确率: {results['precision']*100:.2f}%")
    print(f"加权召回率: {results['recall']*100:.2f}%")
    print(f"加权F1分数: {results['f1_score']*100:.2f}%")
    print(f"\n各类别详细指标:")
    print(f"{'='*60}")
    
    for class_name, metrics in results['per_class'].items():
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall: {metrics['recall']*100:.2f}%")
        print(f"  F1-Score: {metrics['f1_score']*100:.2f}%")
        print(f"  Support: {metrics['support']}")
    
    print(f"\n{'='*60}")
    print("详细分类报告:")
    print(f"{'='*60}")
    print(results['classification_report'])


def main():
    parser = argparse.ArgumentParser(description='评估脑肿瘤分类模型')
    parser.add_argument('--model', type=str, required=True,
                       choices=['pure_cnn', 'pure_vit', 'hybrid', 'parallel', 'embedded'],
                       help='模型类型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_root', type=str,
                       default='/home/runner/work/NN_transformer/NN_transformer/data/raw',
                       help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试集
    print("加载测试集...")
    test_dataset = BrainTumorDataset(
        data_root=args.data_root,
        split='test',
        transform=get_transform(224, 'test')
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    print(f"测试集类别分布: {test_dataset.get_class_distribution()}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    print(f"\n加载模型: {args.model}")
    model = create_model(args.model, num_classes=4)
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"已加载检查点: {args.checkpoint}")
    if 'best_val_acc' in checkpoint:
        print(f"验证集最佳准确率: {checkpoint['best_val_acc']:.2f}%")
    
    # 评估模型
    class_names = BrainTumorDataset.CLASS_NAMES
    results, cm = evaluate_model(model, test_loader, device, class_names)
    
    # 打印结果
    print_results(results, args.model)
    
    # 保存结果
    results_path = output_dir / f'{args.model}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n结果已保存到: {results_path}")
    
    # 绘制混淆矩阵
    cm_path = output_dir / f'{args.model}_confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path)


if __name__ == '__main__':
    main()
