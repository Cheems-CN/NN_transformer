"""
src.models.layers.embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

该模块实现了 Transformer 模型所需的完整嵌入层子系统。

包含以下核心组件：
    1. TokenEmbedding: 负责将离散的 Token ID 映射为稠密向量，并按 sqrt(d_model) 进行缩放。
    2. PositionEncoding: 负责生成正弦/余弦绝对位置编码，并将其注入到 Embedding 中。
    3. TransformerEmbedding: (对外接口) 将上述两者封装，直接将 Input IDs 转换为准备好进入 Attention 层的向量。

Usage:
    from src.models.layers.embeddings import TransformerEmbedding

    # 初始化
    emb_layer = TransformerEmbedding(vocab_size=10000, embedding_dim=512)

    # 前向传播
    x = torch.LongTensor([[1, 2, 3]]) # Batch input
    output = emb_layer(x)
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    TokenEmbedding: 标准的词嵌入层 (Lookup Table + Scaling)。

    该类继承自 nn.Module，负责两件事：
    1. 使用 nn.Embedding 进行查表。
    2. 将结果乘以 sqrt(d_model) 以匹配 Positional Encoding 的量级。

    Args:
        vocab_size (int): 词汇表大小。
        embedding_dim (int): 嵌入向量的维度 (d_model)。
        padding_idx (int, optional): 填充索引，该索引的向量将被初始化为零且不更新梯度。默认 None。

    Attributes:
        embedding (nn.Embedding): 底层的 PyTorch 嵌入层。
        embedding_dim (int): 记录的嵌入维度。
    """

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tokens (torch.Tensor): 输入的 ID 张量。Shape: (batch_size, seq_length)

        Returns:
            torch.Tensor: 缩放后的嵌入向量。Shape: (batch_size, seq_length, embedding_dim)
        """
        output = self.embedding(input_tokens)
        output = output * math.sqrt(self.embedding_dim)
        return output


class PositionEncoding(nn.Module):
    """
    PositionEncoding: 正弦/余弦位置编码注入层。

    该类负责生成固定的位置特征 (不参与训练)，并将其加到输入的 Embedding 向量上，
    最后应用 Dropout 以防止过拟合。

    Args:
        d_model (int): 嵌入维度。
        max_len (int): 预计算的最大序列长度 (默认 5000)。
        dropout (float): Dropout 概率 (默认 0.1)。

    Attributes:
        pe (torch.Tensor): 注册的 buffer，存储预计算的位置编码矩阵。Shape: (1, max_len, d_model)。
        dropout (nn.Dropout): Dropout 层。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 1. 预计算 PE 矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        angle = position * div_term
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)

        # 2. 增加 Batch 维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 3. 注册为 Buffer
        self.register_buffer('position_encoding', pe)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): 输入的 Embedding 张量。Shape: (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: output = input + PE (切片广播) -> Dropout
        """
        # 切片逻辑：只取当前句子长度对应的 PE
        pe_slice = self.position_encoding[:, :input_tensor.size(1), :]
        output = input_tensor + pe_slice
        return self.dropout(output)


class TransformerEmbedding(nn.Module):
    """
    TransformerEmbedding: 嵌入层总封装 (Entry Point)。

    它组合了 TokenEmbedding 和 PositionEncoding，对外提供统一的接口。
    数据流：Input IDs -> [Token Lookup] -> [Scale] -> [Add PE] -> [Dropout] -> Output

    Args:
        vocab_size (int): 词汇表大小 (默认 10000)。
        embedding_dim (int): d_model 维度 (默认 512)。
        padding_idx (int): 填充符 ID (默认 0)。
        max_len (int): 最大序列长度 (默认 5000)。
        dropout (float): Dropout 概率 (默认 0.1)。
    """

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 512, padding_idx: int = 0, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.position_embedding = PositionEncoding(d_model=embedding_dim, max_len=max_len, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入的 Token ID 序列。Shape: (batch_size, seq_length)

        Returns:
            torch.Tensor: 最终的 Transformer 输入向量。Shape: (batch_size, seq_length, embedding_dim)
        """
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        return x