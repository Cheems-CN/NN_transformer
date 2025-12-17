"""
src.models.layers.decoder_layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

实现 Transformer 的解码器层 (Decoder Layer)。
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    DecoderLayer: Transformer 解码器的基本组成单元。

    与 EncoderLayer 不同，它包含三个子层：
    1. Masked Multi-Head Self-Attention: 让解码器“看”自己生成的历史，通过掩码防止看到未来。
    2. Multi-Head Cross-Attention: 让解码器“看”编码器的输出 (K, V)，提取源语言信息。
    3. Position-wise Feed-Forward Network: 逐位置的前馈处理。

    每个子层都遵循 Post-Norm 结构: Output = LayerNorm(x + Dropout(Sublayer(x)))
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): 词嵌入维度 (e.g., 512).
            n_head (int): 注意力头数 (e.g., 8).
            d_ff (int): FFN 隐藏层维度 (e.g., 2048).
            dropout (float): 丢弃率 (default: 0.1).
        """
        super().__init__()

        # 1. 自注意力 (self_attn) - 用于处理 Target 序列内部关系
        self.self_attn = MultiHeadAttention(d_model, n_head)

        # 2. 交叉注意力 (cross_attn) - 用于连接 Source (Encoder) 和 Target (Decoder)
        self.cross_attn = MultiHeadAttention(d_model, n_head)

        # 3. 前馈网络 (feed_forward)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 4. 归一化层 (三个子层各对应一个)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, trg_mask):
        """
        前向传播计算。

        Args:
            x (torch.Tensor): 解码器的输入特征。
                Shape: [batch_size, seq_len_trg, d_model]
            enc_output (torch.Tensor): 编码器的输出特征 (作为 Key 和 Value)。
                Shape: [batch_size, seq_len_src, d_model]
            src_mask (torch.Tensor): 源语言掩码 (用于 Cross-Attention)，屏蔽 Padding。
                Shape: [batch_size, 1, 1, seq_len_src]
            trg_mask (torch.Tensor): 目标语言掩码 (用于 Self-Attention)，屏蔽未来信息 (上三角掩码)。
                Shape: [batch_size, 1, seq_len_trg, seq_len_trg]

        Returns:
            torch.Tensor: 解码器层的输出。
                Shape: [batch_size, seq_len_trg, d_model]
        """

        # --- 子层 1: Masked Self-Attention ---
        residual = x
        _x, _ = self.self_attn(x, x, x, mask=trg_mask)
        x = self.norm1(residual + self.dropout(_x))

        # --- 子层 2: Cross-Attention ---
        residual = x
        _x, _ = self.cross_attn(q=x, k=enc_output, v=enc_output, mask=src_mask)
        x = self.norm2(residual + self.dropout(_x))

        # --- 子层 3: Feed Forward ---
        residual = x
        _x = self.feed_forward(x)
        x = self.norm3(residual + self.dropout(_x))

        return x