"""
src.models.blocks.encoder
~~~~~~~~~~~~~~~~~~~~~~~~~

实现 Transformer 的完整编码器 (Encoder Block)。
"""

import torch
import torch.nn as nn

from ..layers.embeddings import TransformerEmbedding
from ..layers.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    Transformer Encoder Block.

    它负责将输入的索引序列转换为包含上下文信息的稠密向量序列。

    架构流程:
    Input IDs -> Embedding + Positional Encoding -> [EncoderLayer * N] -> LayerNorm -> Output Vectors
    """

    def __init__(self, d_model: int, n_vocab: int, max_len: int, n_layers: int, n_head: int, d_ff: int,
                 dropout: float = 0.1, padding_idx: int = 0):
        """
        初始化编码器。

        Args:
            d_model (int): 词嵌入维度 (e.g., 512).
            n_vocab (int): 词表大小.
            max_len (int): 序列最大长度 (用于位置编码).
            n_layers (int): 堆叠层数 (e.g., 6).
            n_head (int): 注意力头数.
            d_ff (int): FFN 隐藏层维度.
            dropout (float): 丢弃率.
            padding_idx (int): 填充符 ID.
        """
        super().__init__()

        # 1. 基础嵌入组件 (Bug Fix: 必须传入参数)
        self.embedding = TransformerEmbedding(
            vocab_size=n_vocab,
            embedding_dim=d_model,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx
        )

        # 2. 核心堆叠 (ModuleList 是必须的，否则参数无法更新)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 3. 最终归一化 (Standard Post-Norm)
        # 编码器的输出通常会做一次 LayerNorm，以便解码器能够更好地处理
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src=None, vit_input=None, mask=None):
        """
        前向传播逻辑。支持 NLP (src) 和 ViT (vit_input) 两种输入模式。

        Args:
            src (torch.Tensor, optional): 输入序列索引 (NLP模式)。
                Shape: [batch_size, seq_len]
            vit_input (torch.Tensor, optional): 预嵌入的向量序列 (ViT模式)。
                Shape: [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): 填充掩码 (Padding Mask)。
                Shape: [batch_size, 1, 1, seq_len]

        Returns:
            torch.Tensor: 编码器最终的上下文输出。
                Shape: [batch_size, seq_len, d_model]

        Raises:
            ValueError: 当同时传入 src 和 vit_input，或二者都未传入时抛出。
        """

        # --- A. 预处理 (输入路由) ---
        if src is not None and vit_input is not None:
            raise ValueError("冲突: 不可同时传入 'src' 和 'vit_input'。")

        if vit_input is not None:
            # ViT 模式: Bypass embedding layer
            x = vit_input
        elif src is not None:
            # NLP 模式: Standard embedding lookup
            x = self.embedding(src)
        else:
            raise ValueError("缺失: 必须传入 'src' (NLP) 或 'vit_input' (ViT) 其中之一。")

        # --- B. 堆叠循环 (Loop N times) ---
        for layer in self.layers:
            x = layer(x, mask)

        # --- C. 最终出口 ---
        return self.norm(x)