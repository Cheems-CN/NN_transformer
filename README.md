# 脑肿瘤MRI影像分类实验

基于CNN与Transformer融合架构的脑肿瘤多序列MRI影像分类研究

## 项目概述

本项目实现了五种深度学习模型用于脑肿瘤MRI影像的自动分类：
1. **Pure CNN** - 基于ResNet18的纯卷积神经网络
2. **Pure ViT** - 纯视觉Transformer模型
3. **Hybrid Model** - CNN+Transformer串行融合架构
4. **Parallel Model** - CNN和Transformer并行融合架构（新增）
5. **Embedded Model** - CNN中嵌入Transformer注意力模块（新增）

数据集包含4类脑肿瘤影像：
- 胶质瘤（Glioma Tumor）
- 脑膜瘤（Meningioma Tumor）
- 无肿瘤（No Tumor）
- 垂体瘤（Pituitary Tumor）

## 项目结构

```
NN_transformer/
├── data/
│   └── raw/
│       ├── Training/          # 训练集 (2870张)
│       └── Testing/           # 测试集 (394张)
├── src/
│   ├── datasets/
│   │   └── brain_tumor_dataset.py    # 数据加载器
│   ├── models/
│   │   ├── pure_cnn.py               # 纯CNN模型
│   │   ├── vit.py                    # 纯ViT模型
│   │   ├── hybrid_model.py           # 串行混合模型
│   │   ├── parallel_model.py         # 并行融合模型（新增）
│   │   ├── embedded_model.py         # 嵌入式融合模型（新增）
│   │   ├── blocks/                   # Transformer基础模块
│   │   └── layers/                   # 网络层实现
│   └── utils/
├── scripts/
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   ├── run_experiments.py    # 完整实验流程
│   └── quick_demo.py         # 快速演示
├── latex/
│   └── main/
│       ├── paper.tex         # 完整论文
│       ├── intro.tex         # 引言
│       ├── method.tex        # 实验方法
│       ├── results.tex       # 实验结果
│       ├── discussion.tex    # 讨论
│       ├── conclusion.tex    # 结论
│       └── bi.bib            # 参考文献
├── requirements.txt
└── readme.md
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- Pillow >= 9.0.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0

### 2. 数据准备

确保数据集放置在 `data/raw/` 目录下，结构如下：

```
data/raw/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## 使用方法

### 方法1：完整实验流程（推荐）

运行完整的训练-评估-对比流程：

```bash
python scripts/run_experiments.py --epochs 50 --batch_size 32
```

参数说明：
- `--epochs`: 训练轮数（默认50）
- `--batch_size`: 批大小（默认32）
- `--data_root`: 数据根目录（默认data/raw）
- `--skip_training`: 跳过训练，仅评估
- `--skip_evaluation`: 跳过评估，仅训练

### 方法2：单独训练模型

训练单个模型：

```bash
# 训练纯CNN模型
python scripts/train.py --model pure_cnn --epochs 50 --batch_size 32

# 训练纯ViT模型
python scripts/train.py --model pure_vit --epochs 50 --batch_size 32

# 训练串行混合模型
python scripts/train.py --model hybrid --epochs 50 --batch_size 32

# 训练并行融合模型（新增）
python scripts/train.py --model parallel --epochs 50 --batch_size 32

# 训练嵌入式融合模型（新增）
python scripts/train.py --model embedded --epochs 50 --batch_size 32
```

### 方法3：评估已训练模型

```bash
python scripts/evaluate.py \
    --model pure_cnn \
    --checkpoint ./checkpoints/pure_cnn_best.pth \
    --output_dir ./results
```

### 方法4：快速演示（3个epochs）

如果想快速测试整个流程（用于调试或演示）：

```bash
python scripts/quick_demo.py
```

## 输出结果

### 训练输出

训练过程会自动保存在 `./checkpoints/` 目录：
- `{model_name}_best.pth` - 最佳模型权重
- `{model_name}_epoch{N}.pth` - 定期保存的checkpoint
- `{model_name}_history.json` - 训练历史记录
- `{model_name}_curves.png` - 训练曲线图（Loss和Accuracy）

### 评估输出

评估结果保存在 `./results/` 目录：
- `{model_name}_results.json` - 详细评估指标
- `{model_name}_confusion_matrix.png` - 混淆矩阵图
- `model_comparison.csv` - 模型对比表格
- `model_comparison.png` - 整体性能对比图
- `per_class_comparison.png` - 各类别性能对比图
- `experiment_report.txt` - 完整实验报告

## 模型架构说明

### Pure CNN (ResNet18-based)
- 参数量: ~11.18M
- 特点: 局部特征提取能力强，训练稳定
- 适用场景: 注重局部纹理细节的分类任务

### Pure ViT (Vision Transformer)
- 参数量: 根据配置可调（默认配置约30M）
- 特点: 全局建模能力强，捕捉长程依赖
- 适用场景: 需要理解全局结构关系的任务

### Hybrid Model (串行融合)
- 参数量: 根据配置可调（默认配置约25M）
- 特点: CNN先提取局部特征，再通过Transformer建模全局依赖
- 架构: 串行连接，CNN输出作为Transformer输入

架构流程：
```
Input Image
    ↓
CNN Feature Extractor (局部特征)
    ↓
Feature Map → Sequence (序列化)
    ↓
Linear Projection (投影到Transformer维度)
    ↓
Add [CLS] Token + Position Encoding
    ↓
Transformer Encoder (全局建模)
    ↓
Classification Head (分类输出)
```

### Parallel Model (并行融合) - 新增
- 参数量: ~31.36M
- 特点: CNN和Transformer并行处理输入，融合互补特征
- 优势: 同时获取局部和全局信息，特征更加丰富

架构流程：
```
                Input Image
                     |
            +--------+--------+
            ↓                 ↓
    CNN Branch          ViT Branch
  (局部特征提取)      (全局特征提取)
            ↓                 ↓
      [B, 512]           [B, 512]
            |                 |
            +--------+--------+
                     ↓
           Feature Fusion (拼接+投影)
                     ↓
             Classification Head
                     ↓
                  Output
```

### Embedded Model (嵌入式融合) - 新增
- 参数量: ~4.40M
- 特点: 在CNN主干网络中嵌入Transformer注意力模块（MobileViT风格）
- 优势: 参数量小，计算高效，适合移动端部署

架构流程：
```
Input Image
    ↓
Stem Conv (初始卷积)
    ↓
Stage 1 (纯卷积层)
    ↓
Stage 2 (纯卷积层)
    ↓
Stage 3 (Conv + Transformer Block)
    ↓
Stage 4 (Conv + Transformer Block)
    ↓
Global Average Pooling
    ↓
Classification Head
    ↓
Output
```
Classification Head (分类输出)
```

## 评估指标

本项目使用以下指标全面评估模型性能：
- **Accuracy** (准确率): 整体分类正确率
- **Precision** (精确率): 每个类别的预测精度
- **Recall** (召回率): 每个类别的识别率
- **F1-Score**: 精确率和召回率的调和平均
- **Confusion Matrix** (混淆矩阵): 各类别间的分类情况

## 实验配置

默认训练配置：
- 优化器: AdamW (lr=1e-4, weight_decay=0.01)
- 学习率调度: Cosine Annealing
- 批大小: 32
- 训练轮数: 50 epochs
- 损失函数: Cross-Entropy Loss
- 数据增强: 随机翻转、旋转、颜色抖动

## 论文撰写

LaTeX论文模板位于 `latex/main/` 目录：

### 编译论文

```bash
cd latex/main
xelatex paper.tex
bibtex paper
xelatex paper.tex
xelatex paper.tex
```

或使用 pdflatex（可能不支持中文）：
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### 论文结构

- `paper.tex` - 主文档
- `intro.tex` - 引言部分（已完成）
- `method.tex` - 实验方法部分（已完成）
- `results.tex` - 实验结果部分（需填入实验数据）
- `discussion.tex` - 讨论部分（已完成）
- `conclusion.tex` - 结论部分（已完成）
- `bi.bib` - 参考文献库（已完成）

**注意**: `results.tex` 中标记为 `XX.XX` 的部分需要在完成实验后填入实际数据。

## 常见问题

### Q: 训练很慢怎么办？
A: 可以尝试：
1. 减小批大小（但可能影响性能）
2. 使用GPU加速
3. 减少训练轮数（用于快速测试）
4. 使用更小的模型配置

### Q: 内存不足怎么办？
A: 
1. 减小批大小
2. 减少模型层数或隐藏层维度
3. 使用梯度累积

### Q: 如何修改模型配置？
A: 在 `train.py` 中的 `create_model()` 函数内修改模型参数。

### Q: 如何使用自己的数据集？
A: 
1. 按照相同的目录结构组织数据
2. 修改 `brain_tumor_dataset.py` 中的 `CLASS_NAMES`
3. 相应修改模型的 `num_classes` 参数

## 贡献者

本项目是《神经网络与深度学习》课程的课程论文项目。

## 许可证

本项目仅用于学术研究和教学目的。

## 引用

如果本项目对您的研究有帮助，请引用相关论文。

## 更新日志

- 2025-01-03: 初始版本发布
  - 实现三种模型架构
  - 完成训练和评估框架
  - 完成LaTeX论文模板
