"""
完整实验运行脚本
依次训练和评估三种模型，并生成对比结果
"""

import os
import sys
import json
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"执行命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ 命令执行失败，返回码: {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} 完成")
        return True


def train_all_models(data_root, epochs, batch_size):
    """训练所有五种模型"""
    models = ['pure_cnn', 'pure_vit', 'hybrid', 'parallel', 'embedded']
    
    for model_name in models:
        print(f"\n\n{'#'*70}")
        print(f"# 开始训练模型: {model_name.upper()}")
        print(f"{'#'*70}")
        
        cmd = [
            'python3', 'train.py',
            '--model', model_name,
            '--data_root', data_root,
            '--batch_size', str(batch_size),
            '--epochs', str(epochs),
            '--lr', '1e-4',
            '--num_workers', '4',
            '--save_dir', './checkpoints'
        ]
        
        success = run_command(cmd, f"训练 {model_name} 模型")
        
        if not success:
            print(f"⚠️ {model_name} 训练失败，继续下一个模型")


def evaluate_all_models(data_root, batch_size):
    """评估所有模型"""
    models = ['pure_cnn', 'pure_vit', 'hybrid', 'parallel', 'embedded']
    
    for model_name in models:
        checkpoint_path = f'./checkpoints/{model_name}_best.pth'
        
        if not Path(checkpoint_path).exists():
            print(f"⚠️ 未找到 {model_name} 的最佳模型，跳过评估")
            continue
        
        print(f"\n\n{'#'*70}")
        print(f"# 评估模型: {model_name.upper()}")
        print(f"{'#'*70}")
        
        cmd = [
            'python3', 'evaluate.py',
            '--model', model_name,
            '--checkpoint', checkpoint_path,
            '--data_root', data_root,
            '--batch_size', str(batch_size),
            '--num_workers', '4',
            '--output_dir', './results'
        ]
        
        run_command(cmd, f"评估 {model_name} 模型")


def generate_comparison_report():
    """生成对比报告"""
    print(f"\n\n{'#'*70}")
    print(f"# 生成对比报告")
    print(f"{'#'*70}\n")
    
    models = ['pure_cnn', 'pure_vit', 'hybrid', 'parallel', 'embedded']
    results_dir = Path('./results')
    
    # 收集所有模型的结果
    all_results = {}
    for model_name in models:
        result_path = results_dir / f'{model_name}_results.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                all_results[model_name] = json.load(f)
    
    if not all_results:
        print("❌ 未找到任何评估结果")
        return
    
    # 创建对比表格
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append({
            '模型': model_name,
            '准确率 (%)': results['accuracy'] * 100,
            '精确率 (%)': results['precision'] * 100,
            '召回率 (%)': results['recall'] * 100,
            'F1分数 (%)': results['f1_score'] * 100
        })
    
    df = pd.DataFrame(comparison_data)
    
    # 打印表格
    print("\n模型对比结果:")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # 保存为CSV
    csv_path = results_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n对比表格已保存到: {csv_path}")
    
    # 生成对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['准确率 (%)', '精确率 (%)', '召回率 (%)', 'F1分数 (%)']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = df[metric].values
        bars = ax.bar(df['模型'], values, color=colors)
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'模型{metric}对比', fontsize=14, pad=10)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    comparison_plot_path = results_dir / 'model_comparison.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比图表已保存到: {comparison_plot_path}")
    
    # 每个类别的详细对比
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    class_metrics = {cls: {'模型': [], 'Precision': [], 'Recall': [], 'F1-Score': []} 
                     for cls in class_names}
    
    for model_name, results in all_results.items():
        for cls in class_names:
            if cls in results['per_class']:
                class_metrics[cls]['模型'].append(model_name)
                class_metrics[cls]['Precision'].append(results['per_class'][cls]['precision'] * 100)
                class_metrics[cls]['Recall'].append(results['per_class'][cls]['recall'] * 100)
                class_metrics[cls]['F1-Score'].append(results['per_class'][cls]['f1_score'] * 100)
    
    # 绘制每个类别的对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, cls in enumerate(class_names):
        ax = axes[idx // 2, idx % 2]
        
        df_cls = pd.DataFrame(class_metrics[cls])
        
        x = np.arange(len(df_cls['模型']))
        width = 0.25
        
        bars1 = ax.bar(x - width, df_cls['Precision'], width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, df_cls['Recall'], width, label='Recall', color='#e74c3c')
        bars3 = ax.bar(x + width, df_cls['F1-Score'], width, label='F1-Score', color='#2ecc71')
        
        ax.set_ylabel('分数 (%)', fontsize=11)
        ax.set_title(f'{cls} 类别的模型对比', fontsize=13, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(df_cls['模型'])
        ax.legend(fontsize=10)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    per_class_plot_path = results_dir / 'per_class_comparison.png'
    plt.savefig(per_class_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"各类别详细对比图已保存到: {per_class_plot_path}")
    
    # 生成综合报告
    report_path = results_dir / 'experiment_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("脑肿瘤MRI影像分类实验报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("一、实验概述\n")
        f.write("-"*70 + "\n")
        f.write("本实验对比了五种深度学习模型在脑肿瘤MRI影像分类任务上的性能：\n")
        f.write("1. Pure CNN (纯卷积神经网络) - 基于ResNet18架构\n")
        f.write("2. Pure ViT (纯视觉Transformer) - 基于ViT架构\n")
        f.write("3. Hybrid Model (混合模型) - CNN+Transformer串行融合架构\n")
        f.write("4. Parallel Model (并行模型) - CNN和Transformer并行融合架构\n")
        f.write("5. Embedded Model (嵌入式模型) - CNN中嵌入Transformer注意力模块\n\n")
        
        f.write("数据集信息：\n")
        f.write("- 训练集：2870张图像\n")
        f.write("- 测试集：394张图像\n")
        f.write("- 类别：4类（glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor）\n\n")
        
        f.write("二、整体性能对比\n")
        f.write("-"*70 + "\n")
        f.write(df.to_string(index=False) + "\n\n")
        
        # 找出最佳模型
        best_model = df.loc[df['准确率 (%)'].idxmax(), '模型']
        best_acc = df.loc[df['准确率 (%)'].idxmax(), '准确率 (%)']
        
        f.write(f"最佳模型: {best_model} (准确率: {best_acc:.2f}%)\n\n")
        
        f.write("三、各类别详细性能\n")
        f.write("-"*70 + "\n")
        for model_name, results in all_results.items():
            f.write(f"\n{model_name.upper()}:\n")
            for cls in class_names:
                if cls in results['per_class']:
                    metrics = results['per_class'][cls]
                    f.write(f"  {cls}:\n")
                    f.write(f"    Precision: {metrics['precision']*100:.2f}%\n")
                    f.write(f"    Recall: {metrics['recall']*100:.2f}%\n")
                    f.write(f"    F1-Score: {metrics['f1_score']*100:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告生成完成\n")
        f.write("="*70 + "\n")
    
    print(f"实验报告已保存到: {report_path}")
    
    print("\n✅ 对比报告生成完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行完整实验流程')
    parser.add_argument('--data_root', type=str,
                       default='../data/raw',
                       help='数据根目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--skip_training', action='store_true', help='跳过训练，仅评估')
    parser.add_argument('--skip_evaluation', action='store_true', help='跳过评估，仅训练')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("脑肿瘤MRI影像分类实验")
    print("="*70)
    print(f"\n实验配置:")
    print(f"  数据目录: {args.data_root}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批大小: {args.batch_size}")
    print(f"  跳过训练: {args.skip_training}")
    print(f"  跳过评估: {args.skip_evaluation}")
    
    # 创建必要的目录
    Path('./checkpoints').mkdir(exist_ok=True)
    Path('./results').mkdir(exist_ok=True)
    
    # 训练阶段
    if not args.skip_training:
        train_all_models(args.data_root, args.epochs, args.batch_size)
    else:
        print("\n⏩ 跳过训练阶段")
    
    # 评估阶段
    if not args.skip_evaluation:
        evaluate_all_models(args.data_root, args.batch_size)
        generate_comparison_report()
    else:
        print("\n⏩ 跳过评估阶段")
    
    print("\n" + "="*70)
    print("✅ 实验流程全部完成！")
    print("="*70)
    print("\n结果文件:")
    print("  - 训练checkpoints: ./checkpoints/")
    print("  - 评估结果: ./results/")
    print("  - 对比报告: ./results/experiment_report.txt")
    print("  - 对比图表: ./results/model_comparison.png")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
