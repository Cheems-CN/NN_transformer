"""
快速演示脚本 - 使用少量epochs进行快速训练演示
"""

import subprocess
import sys
from pathlib import Path

def run_quick_demo():
    """运行快速演示实验（3个epochs用于演示）"""
    
    print("\n" + "="*70)
    print("快速演示实验 - 使用3个epochs进行训练")
    print("="*70 + "\n")
    
    models = ['pure_cnn', 'pure_vit', 'hybrid']
    epochs = 3
    batch_size = 32
    
    # 创建目录
    Path('./checkpoints').mkdir(exist_ok=True)
    Path('./results').mkdir(exist_ok=True)
    
    for model_name in models:
        print(f"\n{'#'*70}")
        print(f"# 训练模型: {model_name.upper()} (演示模式: {epochs} epochs)")
        print(f"{'#'*70}\n")
        
        # 训练
        cmd = [
            sys.executable, 'scripts/train.py',
            '--model', model_name,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--num_workers', '2',
            '--save_dir', './checkpoints'
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            print(f"⚠️ {model_name} 训练出现问题")
            continue
        
        # 评估
        checkpoint_path = f'./checkpoints/{model_name}_best.pth'
        if not Path(checkpoint_path).exists():
            print(f"⚠️ 未找到 {model_name} 的checkpoint")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# 评估模型: {model_name.upper()}")
        print(f"{'#'*70}\n")
        
        cmd = [
            sys.executable, 'scripts/evaluate.py',
            '--model', model_name,
            '--checkpoint', checkpoint_path,
            '--batch_size', str(batch_size),
            '--num_workers', '2',
            '--output_dir', './results'
        ]
        
        subprocess.run(cmd, capture_output=False)
    
    # 生成对比报告
    print(f"\n{'#'*70}")
    print(f"# 生成对比报告")
    print(f"{'#'*70}\n")
    
    cmd = [sys.executable, '-c', '''
import sys
sys.path.insert(0, ".")
from scripts.run_experiments import generate_comparison_report
generate_comparison_report()
''']
    
    subprocess.run(cmd)
    
    print("\n" + "="*70)
    print("✅ 快速演示实验完成！")
    print("="*70)
    print("\n注意: 这是演示版本（仅3个epochs），实际实验应使用50+ epochs")
    print("\n结果保存在:")
    print("  - checkpoints/")
    print("  - results/")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_quick_demo()
