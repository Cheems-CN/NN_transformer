"""
Test the new parallel and embedded fusion models
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.parallel_model import ParallelModel
from src.models.embedded_model import EmbeddedModel


def test_parallel_model():
    """测试并行融合模型"""
    print("测试并行融合模型...")
    
    model = ParallelModel(
        num_classes=4,
        feature_dim=512,
        vit_d_model=512,
        vit_n_layers=6,
        vit_n_head=8,
        vit_d_ff=2048,
        dropout=0.1
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params / 1e6:.2f}M")
    
    # 测试前向传播
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (batch_size, 4), f"输出形状错误: {output.shape}"
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ 并行融合模型测试通过\n")


def test_embedded_model():
    """测试嵌入式融合模型"""
    print("测试嵌入式融合模型...")
    
    model = EmbeddedModel(
        num_classes=4,
        base_channels=32,
        transformer_channels=256,
        n_head=4,
        n_transformer_layers=2,
        dropout=0.1
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params / 1e6:.2f}M")
    
    # 测试前向传播
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (batch_size, 4), f"输出形状错误: {output.shape}"
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ 嵌入式融合模型测试通过\n")


def test_gradients():
    """测试梯度反向传播"""
    print("测试梯度反向传播...")
    
    # 测试并行模型
    print("  测试并行模型梯度...")
    parallel_model = ParallelModel(num_classes=4, feature_dim=256)
    parallel_model.train()
    
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_labels = torch.randint(0, 4, (2,))
    
    output = parallel_model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_labels)
    loss.backward()
    
    # 检查是否有梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in parallel_model.parameters() if p.requires_grad)
    assert has_grad, "并行模型没有梯度"
    print("    ✅ 并行模型梯度正常")
    
    # 测试嵌入式模型
    print("  测试嵌入式模型梯度...")
    embedded_model = EmbeddedModel(num_classes=4, base_channels=32)
    embedded_model.train()
    
    output = embedded_model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_labels)
    loss.backward()
    
    # 检查是否有梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in embedded_model.parameters() if p.requires_grad)
    assert has_grad, "嵌入式模型没有梯度"
    print("    ✅ 嵌入式模型梯度正常\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("开始测试新增模型")
    print("="*60 + "\n")
    
    test_parallel_model()
    test_embedded_model()
    test_gradients()
    
    print("="*60)
    print("所有测试通过！✅")
    print("="*60 + "\n")
