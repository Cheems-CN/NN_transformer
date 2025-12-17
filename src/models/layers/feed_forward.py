"""
src.models.layers.feed_forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

实现 Transformer 的 Position-wise Feed-Forward Networks (FFN)。
"""
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    点向全连接前馈网络 (Position-wise Feed-Forward Network)。

    它由两个线性变换组成，中间包含一个 ReLU 激活函数。
    公式: FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model (int): 输入和输出的维度 (例如 512)。
        d_ff (int): 内部隐藏层的维度 (例如 2048)。通常是 d_model 的 4 倍。
        dropout (float): 丢弃率。默认 0.1。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()


        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()

        self.dp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。Shape: [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: 输出张量。Shape: [batch_size, seq_len, d_model]
        """

        x = self.w_1(x)
        x = self.act(x)
        x = self.dp(x)
        x = self.w_2(x)

        return x