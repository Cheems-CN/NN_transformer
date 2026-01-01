import torch
import sys
import os

# 确保能找到 src 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.vit import VisionTransformer


def test_vit_pipeline():
    print("🚀 开始 ViT 全流程集成测试...")

    # 1. 配置参数 (模拟 ViT-Base 配置)
    bs, c, h, w = 2, 3, 224, 224
    n_classes = 10
    d_model = 768

    # 2. 实例化模型
    print(f"\n[1/4] 初始化模型 (Input: {h}x{w}, Classes: {n_classes})...")
    model = VisionTransformer(
        img_size=h,
        patch_size=16,
        n_classes=n_classes,
        d_model=d_model,
        n_layers=12,  # 使用深层网络测试显存和计算流
        n_head=12
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ 模型构建成功。总参数量: {total_params / 1e6:.2f}M (约为 86M 则符合 ViT-Base)")

    # 3. 构造虚拟数据
    dummy_input = torch.randn(bs, c, h, w)
    print(f"\n[2/4] 生成输入数据: {dummy_input.shape}")

    # 4. 前向传播 (最关键的一步)
    print(f"\n[3/4] 执行前向传播 (Forward Pass)...")
    try:
        # 这里会触发 PatchEmbed -> Cat CLS -> Add Pos -> Encoder(vit_input) -> Head
        output = model(dummy_input)

        print(f"   ✅ 前向传播成功！无报错。")
        print(f"   输出形状: {output.shape}")

    except RuntimeError as e:
        print(f"   ❌ 运行时错误 (通常是维度不匹配): {e}")
        return
    except ValueError as e:
        print(f"   ❌ 参数错误 (可能是 Encoder 接口问题): {e}")
        return

    # 5. 维度校验
    print(f"\n[4/4] 验证输出维度...")
    expected_shape = (bs, n_classes)

    if output.shape == expected_shape:
        print(f"   ✅ 维度匹配: {output.shape} == {expected_shape}")
        print("\n🎉 测试通过！你的 ViT 模型已具备训练能力。")
    else:
        print(f"   ❌ 维度不匹配: 期望 {expected_shape}, 实际 {output.shape}")


if __name__ == "__main__":
    test_vit_pipeline()