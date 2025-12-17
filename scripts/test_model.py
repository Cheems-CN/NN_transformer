import torch
from src.models.transformer import Transformer


def test_transformer_forward():
    # 1. 定义超参数 (假装我们有一个小模型)
    src_vocab_size = 100
    trg_vocab_size = 100
    src_max_len = 20
    trg_max_len = 20
    d_model = 512

    # 2. 实例化模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_max_len=src_max_len,
        trg_max_len=trg_max_len,
        d_model=d_model,
        n_layers=2,  # 测试用，2层够了
        n_head=8,
        d_ff=1024,
        dropout=0.1,
        padding_idx=0
    )

    # 3. 构造假数据 (Batch Size = 2)
    # 假设 0 是 Padding，我们随机生成一些 1~99 的索引
    src = torch.randint(1, src_vocab_size, (2, 15))  # 长度 15
    trg = torch.randint(1, trg_vocab_size, (2, 18))  # 长度 18

    # 4. 手动增加 Padding (测试 Mask 是否工作)
    # 把 src 的后 5 位变成 0
    src[:, -5:] = 0

    print("--- 模型结构检查 ---")
    # print(model) # 如果想看详细层级可以取消注释

    print("\n--- 前向传播测试 ---")
    print(f"输入 Src 形状: {src.shape}")
    print(f"输入 Trg 形状: {trg.shape}")

    # 5. 核心：运行 Forward
    try:
        output = model(src, trg)
        print("✅ 模型运行成功！")
        print(f"输出形状: {output.shape}")

        # 6. 验证形状
        expected_shape = (2, 18, trg_vocab_size)
        assert output.shape == expected_shape, f"形状错误！期望 {expected_shape}, 实际 {output.shape}"
        print("✅ 输出形状验证通过！")

    except Exception as e:
        print("❌ 模型运行失败！")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_transformer_forward()