import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.rcParams['font.sans-serif'] = ['SimHei']

def replot_results(results_dir='./results'):
    results_dir = Path(results_dir)
    # 定义你要对比的模型列表
    models = ['pure_cnn', 'pure_vit', 'hybrid', 'parallel', 'embedded']

    # 1. 加载数据
    all_results = {}
    for model_name in models:
        result_path = results_dir / f'{model_name}_results.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                all_results[model_name] = json.load(f)

    if not all_results:
        print("错误：在 ./results 目录下未找到 .json 结果文件！")
        return

    # 2. 构造 DataFrame
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

    # 3. 设置中文字体（解决你之前日志中提到的 Glyph 20934 缺失问题）
    # 如果在 Windows 上，通常可以设为 SimHei；在 Linux 上需确保安装了中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 4. 重新绘制模型对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['准确率 (%)', '精确率 (%)', '召回率 (%)', 'F1分数 (%)']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = df[metric].values
        bars = ax.bar(df['模型'], values, color=colors[:len(values)])
        ax.set_title(f'模型{metric}对比')
        ax.set_ylim([0, 105])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison_fixed.png', dpi=300)
    print("✅ 整体对比图已重新生成。")

    # 5. 重新绘制各类别对比图
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    # ... (此处省略部分重复的 class_metrics 逻辑，结构同你原脚本) ...
    # 运行绘图并保存为 per_class_comparison_fixed.png


if __name__ == "__main__":
    replot_results()