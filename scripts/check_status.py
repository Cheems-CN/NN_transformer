"""
项目完成情况检查脚本
快速检查所有必需文件和代码是否就绪
"""

import os
from pathlib import Path
import sys

def check_file_exists(path, description):
    """检查文件是否存在"""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}未找到: {path}")
        return False

def check_directory_exists(path, description):
    """检查目录是否存在"""
    if Path(path).is_dir():
        file_count = len(list(Path(path).glob('*')))
        print(f"✅ {description}: {path} ({file_count} 个文件)")
        return True
    else:
        print(f"❌ {description}未找到: {path}")
        return False

def main():
    print("\n" + "="*70)
    print("脑肿瘤MRI影像分类项目 - 完成情况检查")
    print("="*70 + "\n")
    
    all_good = True
    
    # 1. 检查数据集
    print("📊 检查数据集...")
    all_good &= check_directory_exists("data/raw/Training", "训练集")
    all_good &= check_directory_exists("data/raw/Testing", "测试集")
    print()
    
    # 2. 检查模型实现
    print("🤖 检查模型实现...")
    all_good &= check_file_exists("src/models/pure_cnn.py", "纯CNN模型")
    all_good &= check_file_exists("src/models/vit.py", "纯ViT模型")
    all_good &= check_file_exists("src/models/hybrid_model.py", "混合模型")
    all_good &= check_file_exists("src/datasets/brain_tumor_dataset.py", "数据加载器")
    print()
    
    # 3. 检查训练和评估脚本
    print("🔧 检查训练和评估脚本...")
    all_good &= check_file_exists("scripts/train.py", "训练脚本")
    all_good &= check_file_exists("scripts/evaluate.py", "评估脚本")
    all_good &= check_file_exists("scripts/run_experiments.py", "完整实验脚本")
    all_good &= check_file_exists("scripts/quick_demo.py", "快速演示脚本")
    print()
    
    # 4. 检查LaTeX论文
    print("📝 检查LaTeX论文...")
    all_good &= check_file_exists("latex/main/paper.tex", "主文档")
    all_good &= check_file_exists("latex/main/intro.tex", "引言")
    all_good &= check_file_exists("latex/main/method.tex", "实验方法")
    all_good &= check_file_exists("latex/main/results.tex", "实验结果")
    all_good &= check_file_exists("latex/main/discussion.tex", "讨论")
    all_good &= check_file_exists("latex/main/conclusion.tex", "结论")
    all_good &= check_file_exists("latex/main/bi.bib", "参考文献")
    print()
    
    # 5. 检查文档
    print("📚 检查文档...")
    all_good &= check_file_exists("README.md", "README文档")
    all_good &= check_file_exists("EXPERIMENT_GUIDE.md", "实验指南")
    all_good &= check_file_exists("requirements.txt", "依赖列表")
    print()
    
    # 6. 检查模型是否可以导入
    print("🔍 检查模型导入...")
    try:
        sys.path.insert(0, '.')
        from src.models.pure_cnn import PureCNN
        from src.models.vit import VisionTransformer
        from src.models.hybrid_model import HybridModel
        print("✅ 所有模型可以成功导入")
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        all_good = False
    print()
    
    # 7. 快速测试模型
    print("🧪 快速测试模型...")
    try:
        import torch
        
        # 测试纯CNN
        model_cnn = PureCNN(num_classes=4)
        x = torch.randn(1, 3, 224, 224)
        y = model_cnn(x)
        assert y.shape == (1, 4), "CNN输出形状错误"
        print("✅ 纯CNN模型测试通过")
        
        # 测试混合模型
        model_hybrid = HybridModel(num_classes=4)
        y = model_hybrid(x)
        assert y.shape == (1, 4), "混合模型输出形状错误"
        print("✅ 混合模型测试通过")
        
        print("✅ 所有模型功能正常")
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        all_good = False
    print()
    
    # 8. 总结
    print("="*70)
    if all_good:
        print("🎉 项目检查完成！所有组件就绪！")
        print("\n下一步操作：")
        print("1. 运行实验: python scripts/run_experiments.py --epochs 50")
        print("   或快速测试: python scripts/quick_demo.py")
        print("2. 查看结果: 检查 checkpoints/ 和 results/ 目录")
        print("3. 更新论文: 将实验数据填入 latex/main/results.tex")
        print("4. 编译论文: cd latex/main && xelatex paper.tex")
        print("\n详细说明请查看 EXPERIMENT_GUIDE.md")
    else:
        print("⚠️ 部分组件缺失或有问题，请检查上述输出")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
