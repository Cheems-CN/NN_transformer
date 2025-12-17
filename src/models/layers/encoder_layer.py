"""
src.models.layers.encoder_layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

实现 Transformer 的编码器层 (Encoder Layer)。
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    EncoderLayer: Transformer 编码器的基本组成单元。

    它包含两个子层：
    1. Multi-Head Attention Mechanism
    2. Position-wise Feed-Forward Network

    每个子层周围都有残差连接 (Residual Connection) 和层归一化 (Layer Normalization)。
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout=0.1):
        super().__init__()

        # 1. 自注意力层
        self.self_attn = MultiHeadAttention(d_model, n_head)

        # 2. 前馈网络层
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 3. 两个 LayerNorm 层 (分别用于两个子层后面)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播逻辑。

        Args:
            x (torch.Tensor): 输入特征。
                Shape: [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): 掩码张量，用于在自注意力中屏蔽特定位置。
                Shape: [batch_size, 1, seq_len, seq_len] 或 [batch_size, 1, 1, seq_len]

        Returns:
            torch.Tensor: 编码器层的输出。
                Shape: [batch_size, seq_len, d_model] (与输入形状保持一致，以便堆叠)
        """

        residual = x

        attn_output, _ = self.self_attn(x, x, x, mask)
        attn_output = self.dropout(attn_output)  # 2. Dropout

        x = self.norm1(residual + attn_output)  # 3. Add & Norm

        residual = x

        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)  # 2. Dropout

        x = self.norm2(residual + ff_output)  # 3. Add & Norm

        return x