"""
src.models.layers.attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

该模块实现了 Transformer 模型的核心注意力机制。

包含以下类：
    - ScaledDotProductAttention: 计算 Q, K, V 之间的缩放点积注意力。
"""

import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    ScaledDotProductAttention: 标准的缩放点积注意力机制计算单元。

    它执行以下核心公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    该模块没有可学习的参数(Weights)，仅包含 Dropout 层。

    Args:
        dropout (float): 对注意力权重应用的 Dropout 概率。默认 0.1。

    Attributes:
        dropout (nn.Dropout): Dropout 层。
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播计算。

        Args:
            q (torch.Tensor): 查询向量。Shape: (batch_size, n_heads, seq_len_q, d_k)
            k (torch.Tensor): 键向量。  Shape: (batch_size, n_heads, seq_len_k, d_k)
            v (torch.Tensor): 值向量。  Shape: (batch_size, n_heads, seq_len_v, d_v)
            mask (torch.Tensor, optional): 掩码张量。
                                         Shape: (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, seq_len, seq_len)
                                         mask == 0 的位置会被填充为 -1e9。

        Returns:
            tuple:
                - context (torch.Tensor): 加权后的输出向量。Shape: (batch_size, n_heads, seq_len_q, d_v)
                - attn (torch.Tensor): 注意力权重分布图。Shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        """

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 将 mask 为 0 的位置填入极小值 (-1e9)，Softmax 后变为 0
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn