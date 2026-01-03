# 新增模型实现总结

## 概述

根据实验需求，我们在原有的三个模型（Pure CNN、Pure ViT、Hybrid串行融合）基础上，新增了两个融合策略的模型：

1. **Parallel Model（并行融合模型）**
2. **Embedded Model（嵌入式融合模型）**

## 实现的模型

### 1. Parallel Model (并行融合模型)

**文件位置**: `src/models/parallel_model.py`

**架构设计**:
- CNN分支和ViT分支并行处理相同的输入图像
- CNN分支基于ResNet架构，提取局部纹理特征
- ViT分支基于Vision Transformer，提取全局结构特征
- 两个分支的输出特征向量进行拼接（concatenation）
- 通过融合层（多层MLP）整合特征
- 最后通过分类头输出预测结果

**参数配置**:
- 参数量: ~31.36M
- CNN输出维度: 512
- ViT输出维度: 512
- 融合后维度: 1024 → 512 → 256
- ViT配置: 6层，8头注意力

**优势**:
- 同时获取局部和全局信息
- 特征互补性强
- 适合需要多尺度特征的任务

**架构流程**:
```
            Input Image
                 |
        +--------+--------+
        ↓                 ↓
   CNN Branch        ViT Branch
 (局部特征)         (全局特征)
        ↓                 ↓
    [B, 512]          [B, 512]
        |                 |
        +--------+--------+
                 ↓
      Concatenate [B, 1024]
                 ↓
         Fusion Layer
                 ↓
       Classification Head
                 ↓
             Output [B, 4]
```

### 2. Embedded Model (嵌入式融合模型)

**文件位置**: `src/models/embedded_model.py`

**架构设计**:
- 采用MobileViT风格的设计
- 前两个stage使用纯卷积层进行特征提取
- 后两个stage在卷积中嵌入Transformer注意力模块
- MobileViT块结构: 局部卷积 → 序列展开 → Transformer → 序列重组 → 卷积投影
- 通过残差连接保持特征流动

**参数配置**:
- 参数量: ~4.40M（最轻量）
- 基础通道数: 32
- Transformer隐藏维度: 256
- 注意力头数: 4
- 每个MobileViT块包含2层Transformer

**优势**:
- 参数量小，计算效率高
- 适合资源受限场景（如移动端部署）
- 在保持轻量级的同时获得全局建模能力

**架构流程**:
```
Input Image [B, 3, 224, 224]
    ↓
Stem Conv [B, 32, 112, 112]
    ↓
Stage 1 (纯卷积) [B, 64, 56, 56]
    ↓
Stage 2 (纯卷积) [B, 128, 28, 28]
    ↓
Stage 3 (Conv + Transformer) [B, 192, 14, 14]
    ↓
Stage 4 (Conv + Transformer) [B, 256, 7, 7]
    ↓
Global Average Pooling [B, 256]
    ↓
Classification Head [B, 4]
```

## 集成情况

### 训练脚本 (scripts/train.py)
- ✅ 已添加 `parallel` 和 `embedded` 模型选项
- ✅ 已配置合适的超参数
- ✅ 支持命令行参数: `--model parallel` 或 `--model embedded`

### 评估脚本 (scripts/evaluate.py)
- ✅ 已添加模型加载支持
- ✅ 支持评估新模型的性能

### 实验脚本 (scripts/run_experiments.py)
- ✅ 已更新为训练和评估5个模型
- ✅ 报告生成包含所有5个模型的对比

## 使用方法

### 训练新模型

```bash
# 训练并行融合模型
python scripts/train.py --model parallel --epochs 50 --batch_size 32

# 训练嵌入式融合模型
python scripts/train.py --model embedded --epochs 50 --batch_size 32
```

### 评估新模型

```bash
# 评估并行融合模型
python scripts/evaluate.py \
    --model parallel \
    --checkpoint ./checkpoints/parallel_best.pth \
    --output_dir ./results

# 评估嵌入式融合模型
python scripts/evaluate.py \
    --model embedded \
    --checkpoint ./checkpoints/embedded_best.pth \
    --output_dir ./results
```

### 运行完整实验

```bash
# 训练、评估所有5个模型并生成对比报告
python scripts/run_experiments.py --epochs 50 --batch_size 32
```

## 测试验证

创建了专门的测试文件 `tests/test_new_models.py`，包含：

1. **前向传播测试**: 验证模型能够正确处理输入并输出正确形状
2. **参数量统计**: 确认模型参数规模
3. **梯度反向传播测试**: 验证模型能够正确计算梯度
4. **输入输出维度测试**: 确保维度匹配

运行测试：
```bash
python tests/test_new_models.py
```

所有测试均已通过 ✅

## 模型对比

| 模型 | 参数量 | 融合方式 | 特点 |
|------|--------|---------|------|
| Pure CNN | ~11.18M | - | 局部特征提取强，训练稳定 |
| Pure ViT | ~30M | - | 全局建模能力强 |
| Hybrid (串行) | ~25M | CNN→Transformer | 串行处理，先局部后全局 |
| Parallel (并行) | **31.36M** | CNN∥Transformer | 同时提取局部和全局特征 |
| Embedded (嵌入) | **4.40M** | CNN⊕Transformer | 轻量高效，适合移动端 |

## 预期性能

基于架构设计，预期实验结果：

1. **Parallel Model**: 
   - 由于同时利用了CNN的局部特征和ViT的全局特征，预期性能最好
   - 可能在各类别上表现更均衡
   - 预期准确率: 95-97%+

2. **Embedded Model**:
   - 虽然参数量最小，但通过嵌入Transformer获得了全局建模能力
   - 在保持效率的同时应该有不错的性能
   - 预期准确率: 94-96%

## 文档更新

- ✅ `README.md`: 更新了项目概述、模型架构说明、使用方法
- ✅ `EXPERIMENT_GUIDE.md`: 更新了实验流程、预期结果
- ✅ 所有相关文档均已同步更新

## 下一步

1. **运行实验**: 使用 `python scripts/run_experiments.py` 训练所有模型
2. **分析结果**: 查看 `results/experiment_report.txt` 中的对比结果
3. **更新论文**: 将实验结果填入LaTeX论文模板
4. **生成图表**: 使用生成的混淆矩阵和对比图表

## 注意事项

1. **内存使用**: 并行模型参数量较大(31M)，如果内存不足可以减小batch_size
2. **训练时间**: 嵌入式模型虽然参数少，但Transformer计算可能需要更多时间
3. **数据增强**: 所有模型都使用相同的数据增强策略以确保公平对比
4. **随机性**: 设置随机种子可以提高结果可重复性

## 技术细节

### Parallel Model 关键实现
- CNN分支: 基于ResNet的渐进式特征提取
- ViT分支: 标准ViT架构，包含patch embedding和position encoding
- 融合策略: 特征拼接 + 多层感知机降维

### Embedded Model 关键实现
- MobileViT块: 局部卷积→展平→Transformer→重组→投影
- 残差连接: 保持特征流动，避免梯度消失
- 分阶段设计: 前期纯卷积，后期嵌入Transformer

---

**实现完成时间**: 2026-01-03
**实现者**: GitHub Copilot Agent
**状态**: ✅ 完成并测试通过
