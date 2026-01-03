# 实验完成指南

## 项目完成情况

### ✅ 已完成的工作

1. **完整的代码实现**
   - ✅ 五种模型架构（Pure CNN、Pure ViT、Hybrid、Parallel、Embedded）
   - ✅ 数据加载和预处理pipeline
   - ✅ 统一的训练框架
   - ✅ 完整的评估和对比系统
   - ✅ 自动化图表生成

2. **LaTeX论文**
   - ✅ 引言部分（intro.tex）- 已完成
   - ✅ 实验方法部分（method.tex）- 已完成
   - ✅ 实验结果部分（results.tex）- 模板完成，需填充数据
   - ✅ 讨论部分（discussion.tex）- 已完成
   - ✅ 结论部分（conclusion.tex）- 已完成
   - ✅ 参考文献（bi.bib）- 已完成
   - ✅ 主文档（paper.tex）- 已完成

3. **文档和工具**
   - ✅ 完整的README.md使用说明
   - ✅ 快速演示脚本
   - ✅ 完整实验流程脚本

## 🚀 如何完成实验

### 方案1：完整实验（推荐用于正式论文）

运行50个epochs的完整训练：

```bash
# 进入项目目录
cd /home/runner/work/NN_transformer/NN_transformer

# 运行完整实验（大约需要数小时，取决于硬件）
python scripts/run_experiments.py --epochs 50 --batch_size 32
```

### 方案2：快速演示（用于测试和快速验证）

运行3个epochs的快速实验：

```bash
python scripts/quick_demo.py
```

### 方案3：分步骤执行

如果想分别训练每个模型：

```bash
# 1. 训练纯CNN模型
python scripts/train.py --model pure_cnn --epochs 50 --batch_size 32

# 2. 训练纯ViT模型  
python scripts/train.py --model pure_vit --epochs 50 --batch_size 32

# 3. 训练混合模型（串行融合）
python scripts/train.py --model hybrid --epochs 50 --batch_size 32

# 4. 训练并行模型（并行融合）- 新增
python scripts/train.py --model parallel --epochs 50 --batch_size 32

# 5. 训练嵌入式模型（嵌入式融合）- 新增
python scripts/train.py --model embedded --epochs 50 --batch_size 32

# 6. 评估所有模型
python scripts/evaluate.py --model pure_cnn --checkpoint ./checkpoints/pure_cnn_best.pth
python scripts/evaluate.py --model pure_vit --checkpoint ./checkpoints/pure_vit_best.pth
python scripts/evaluate.py --model hybrid --checkpoint ./checkpoints/hybrid_best.pth
python scripts/evaluate.py --model parallel --checkpoint ./checkpoints/parallel_best.pth
python scripts/evaluate.py --model embedded --checkpoint ./checkpoints/embedded_best.pth

# 7. 生成对比报告
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_experiments import generate_comparison_report
generate_comparison_report()
"
```

## 📊 实验结果的使用

### 1. 查看训练过程

实验完成后，检查 `checkpoints/` 目录：

```
checkpoints/
├── pure_cnn_best.pth              # 最佳模型权重
├── pure_cnn_history.json          # 训练历史
├── pure_cnn_curves.png            # 训练曲线
├── pure_vit_best.pth
├── pure_vit_history.json
├── pure_vit_curves.png
├── hybrid_best.pth
├── hybrid_history.json
├── hybrid_curves.png
├── parallel_best.pth              # 并行模型（新增）
├── parallel_history.json
├── parallel_curves.png
├── embedded_best.pth              # 嵌入式模型（新增）
├── embedded_history.json
└── embedded_curves.png
```

### 2. 查看评估结果

检查 `results/` 目录：

```
results/
├── pure_cnn_results.json           # 详细指标
├── pure_cnn_confusion_matrix.png   # 混淆矩阵
├── pure_vit_results.json
├── pure_vit_confusion_matrix.png
├── hybrid_results.json
├── hybrid_confusion_matrix.png
├── parallel_results.json           # 并行模型（新增）
├── parallel_confusion_matrix.png
├── embedded_results.json           # 嵌入式模型（新增）
├── embedded_confusion_matrix.png
├── model_comparison.csv            # 对比表格
├── model_comparison.png            # 整体对比图
├── per_class_comparison.png        # 各类别对比图
└── experiment_report.txt           # 完整报告
```

### 3. 更新论文

打开 `latex/main/results.tex`，找到所有 `XX.XX` 标记，替换为实际数值。

例如，从 `results/model_comparison.csv` 中获取数据：

```csv
模型,准确率 (%),精确率 (%),召回率 (%),F1分数 (%)
pure_cnn,95.43,95.21,95.43,95.30
pure_vit,96.19,96.05,96.19,96.11
hybrid,97.21,97.15,97.21,97.18
parallel,XX.XX,XX.XX,XX.XX,XX.XX
embedded,XX.XX,XX.XX,XX.XX,XX.XX
```

将 `results.tex` 中的表格更新为：

```latex
\begin{tabular}{lcccc}
\hline
\textbf{模型} & \textbf{准确率(\%)} & \textbf{精确率(\%)} & \textbf{召回率(\%)} & \textbf{F1分数(\%)} \\
\hline
Pure CNN & 95.43 & 95.21 & 95.43 & 95.30 \\
Pure ViT & 96.19 & 96.05 & 96.19 & 96.11 \\
Hybrid (Serial) & 97.21 & 97.15 & 97.21 & 97.18 \\
Parallel Fusion & XX.XX & XX.XX & XX.XX & XX.XX \\
Embedded Fusion & XX.XX & XX.XX & XX.XX & XX.XX \\
\hline
\end{tabular}
```

### 4. 插入图表

在 `results.tex` 中取消注释图片插入命令：

```latex
% 原来：
% \includegraphics[width=0.32\textwidth]{results/pure_cnn_confusion_matrix.png}

% 改为：
\includegraphics[width=0.32\textwidth]{../../results/pure_cnn_confusion_matrix.png}
```

## 📝 编译论文

### 使用XeLaTeX（推荐，支持中文）

```bash
cd latex/main
xelatex paper.tex
bibtex paper
xelatex paper.tex
xelatex paper.tex
```

### 使用PDFLaTeX（可能需要额外配置中文支持）

```bash
cd latex/main
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

编译成功后，会生成 `paper.pdf` 文件。

## 🎯 预期结果

基于不同融合策略的设计理念，预期实验结果应该显示：

1. **Pure CNN**
   - 优势：训练快速，参数少，局部特征提取强
   - 准确率：~93-95%
   - 特点：在边缘清晰的类别（如脑膜瘤）表现好

2. **Pure ViT**
   - 优势：全局建模能力强
   - 准确率：~94-96%
   - 特点：可能在训练初期较慢，但最终性能较好

3. **Hybrid Model (串行融合)**
   - 优势：先提取局部特征，再建模全局依赖
   - 准确率：~94-96%
   - 特点：理论上结合两者优势，但实际可能受限于串行结构

4. **Parallel Model (并行融合)** - 新增
   - 优势：同时提取局部和全局特征，特征更加丰富
   - 准确率：预期~95-97%+
   - 特点：参数量较大(31M)，但特征互补性强，可能优于串行

5. **Embedded Model (嵌入式融合)** - 新增
   - 优势：参数量小(4.4M)，效率高，适合移动端
   - 准确率：预期~94-96%
   - 特点：在保持较小模型的同时获得Transformer的全局建模能力

## ⚠️ 注意事项

1. **训练时间**
   - 完整50 epochs可能需要数小时（取决于GPU/CPU）
   - 可以先用quick_demo测试（3 epochs约30-60分钟）

2. **内存需求**
   - 如果出现内存不足，减小batch_size
   - ViT模型通常需要更多内存

3. **随机性**
   - 深度学习训练有随机性，多次运行结果可能略有差异
   - 使用random seed可以提高可重复性

4. **数据增强**
   - 已自动启用（随机翻转、旋转、颜色抖动）
   - 有助于提高泛化能力

## 🔧 故障排除

### 问题1：找不到模块

```bash
# 确保在项目根目录运行
cd /home/runner/work/NN_transformer/NN_transformer
python scripts/train.py ...
```

### 问题2：CUDA out of memory

```bash
# 减小批大小
python scripts/train.py --model hybrid --batch_size 16  # 或更小
```

### 问题3：训练中断

- 检查checkpoints目录是否有保存的模型
- 可以从最近的checkpoint继续训练（需要修改代码加载checkpoint）

### 问题4：论文编译错误

- 确保安装了完整的LaTeX发行版（TeX Live或MiKTeX）
- 确保安装了中文字体支持
- 检查所有图片路径是否正确

## 📚 进一步改进方向

如果时间允许，可以考虑：

1. **数据增强优化**
   - 尝试更多增强策略
   - 使用AutoAugment

2. **超参数调优**
   - 学习率网格搜索
   - 不同的优化器配置

3. **模型集成**
   - 组合三个模型的预测
   - 可能进一步提升性能

4. **可解释性分析**
   - 可视化注意力权重
   - Grad-CAM热图

5. **迁移学习**
   - 使用预训练权重
   - 可能显著提升性能

## 📞 联系方式

如有问题，请参考README.md或查看代码注释。

## ✨ 总结

所有核心工作已完成！只需：
1. ✅ 运行实验脚本
2. ✅ 将结果数据填入论文
3. ✅ 编译LaTeX生成PDF
4. ✅ 提交完整论文

祝实验顺利！🎉
