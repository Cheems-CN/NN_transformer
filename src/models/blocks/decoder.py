"""
src.models.blocks.decoder
~~~~~~~~~~~~~~~~~~~~~~~~~

实现 Transformer 的完整解码器 (Decoder Block)。
"""

import torch
import torch.nn as nn

# 复用之前封装好的 Embedding 组件
from ..layers.embeddings import TransformerEmbedding
# 引入刚才写好的 DecoderLayer
from ..layers.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
    Transformer Decoder Block.

    它负责接收目标序列输入，结合编码器的上下文信息，逐层提取特征。
    整体数据流: Input IDs -> Embedding -> [DecoderLayer * N] -> LayerNorm -> Output Vectors

    Args:
        d_model (int): 模型的维度 (e.g., 512).
        n_vocab (int): 目标语言词表大小.
        max_len (int): 最大序列长度.
        n_layers (int): 堆叠的解码器层数 (e.g., 6).
        n_head (int): 注意力头数.
        d_ff (int): FFN 隐藏层维度.
        dropout (float): 丢弃率.
        padding_idx (int): 填充符 ID.
    """

    def __init__(self, d_model: int, n_vocab: int, max_len: int, n_layers: int, n_head: int, d_ff: int,
                 dropout: float = 0.1, padding_idx: int = 0):
        super().__init__()

        # 1. 嵌入层 (Embedding + Positional Encoding)
        self.embedding = TransformerEmbedding(
            vocab_size=n_vocab,
            embedding_dim=d_model,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx
        )

        # 2. 堆叠 N 层解码器层
        # 使用 ModuleList 以便 PyTorch 追踪参数
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 3. 最终归一化 (Standard Post-Norm)
        # 解码器输出前通常会做一次 LayerNorm，使分布稳定
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, enc_output, src_mask, trg_mask):
        """
        前向传播逻辑。

        Args:
            trg (torch.Tensor): 目标序列索引。
                Shape: [batch, seq_len_trg]
            enc_output (torch.Tensor): 编码器输出 (作为 Key/Value)。
                Shape: [batch, seq_len_src, d_model]
            src_mask (torch.Tensor): 源语言掩码 (屏蔽 enc_output 的 padding)。
                Shape: [batch, 1, 1, seq_len_src]
            trg_mask (torch.Tensor): 目标语言掩码 (屏蔽未来信息)。
                Shape: [batch, 1, seq_len_trg, seq_len_trg]

        Returns:
            torch.Tensor: 解码器最终输出向量。
                Shape: [batch, seq_len_trg, d_model]
        """

        # 1. 预处理: Index -> Vector
        x = self.embedding(trg)

        # 2. 核心堆叠循环
        # 关键点: 每一层都要接收 enc_output (K,V) 和两个 mask
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask)

        # 3. 最终出口
        return self.norm(x)