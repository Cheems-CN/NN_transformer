import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.rcParams['font.sans-serif'] = ['SimHei']

def draw_per_class_comparison(results_dir='./results'):
    results_dir = Path(results_dir)
    # 定义你要对比的模型和类别（请务必检查json里的名称是否一致）
    models = ['pure_cnn', 'pure_vit', 'hybrid']  # 如果有 parallel 和 embedded 也加上
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    # 收集数据
    all_results = {}
    for model_name in models:
        result_path = results_dir / f'{model_name}_results.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                all_results[model_name] = json.load(f)

    if not all_results:
        print("❌ 未找到结果文件，请检查 ./results 目录")
        return

    # 构造绘图数据结构
    class_metrics = {cls: {'模型': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
                     for cls in class_names}

    for model_name in models:
        if model_name in all_results:
            res = all_results[model_name]
            for cls in class_names:
                # 注意：这里要确保 json 里的 key 确实是这些名字
                if 'per_class' in res and cls in res['per_class']:
                    m = res['per_class'][cls]
                    class_metrics[cls]['模型'].append(model_name)
                    class_metrics[cls]['Precision'].append(m['precision'] * 100)
                    class_metrics[cls]['Recall'].append(m['recall'] * 100)
                    class_metrics[cls]['F1-Score'].append(m['f1_score'] * 100)

    # 开始绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows中文支持
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, cls in enumerate(class_names):
        ax = axes[idx // 2, idx % 2]
        df_cls = pd.DataFrame(class_metrics[cls])

        if df_cls.empty: continue

        x = np.arange(len(df_cls['模型']))
        width = 0.25

        bars1 = ax.bar(x - width, df_cls['Precision'], width, label='精确率', color='#3498db')
        bars2 = ax.bar(x, df_cls['Recall'], width, label='召回率', color='#e74c3c')
        bars3 = ax.bar(x + width, df_cls['F1-Score'], width, label='F1分数', color='#2ecc71')

        ax.set_title(f'类别: {cls}', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(df_cls['模型'])
        ax.set_ylim([0, 110])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.1f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = results_dir / 'per_class_comparison_fixed.png'
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ 类别对比图已生成: {save_path}")


if __name__ == "__main__":
    draw_per_class_comparison()