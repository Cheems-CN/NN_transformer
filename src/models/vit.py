"""
Vision Transformer (ViT) 模型主体实现模块。

本模块实现了 Vision Transformer 架构，支持将图像切分为 Patch 序列，
结合 Learnable Position Embedding 和 Class Token，利用 Transformer Encoder
进行图像分类任务。

参考文献:
    Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021.
    https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn

from src.models.layers.patch_embedding import PatchEmbedding
from src.models.blocks.encoder import Encoder


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 用于图像分类任务。

    该模型包含以下主要组件：
    1. PatchEmbedding: 将图像转换为序列。
    2. Class Token: 用于聚合全局特征的可学习向量。
    3. Position Embedding: 可学习的绝对位置编码。
    4. Transformer Encoder: 提取特征的主干网络。
    5. MLP Head: 用于最终分类的线性层。

    Args:
        img_size (int): 输入图像的高度/宽度 (默认: 224)。
        patch_size (int): Patch 的高度/宽度 (默认: 16)。
        in_chans (int): 输入图像的通道数 (默认: 3)。
        n_classes (int): 分类任务的类别数量 (默认: 1000)。
        d_model (int): Transformer 隐藏层维度 (默认: 768)。
        n_layers (int): Transformer Encoder 层数 (默认: 12)。
        n_head (int): 多头注意力的头数 (默认: 12)。
        d_ff (int): FFN 中间层的维度 (默认: 3072)。
        dropout (float): Dropout 比率 (默认: 0.1)。

    Attributes:
        patch_embed (PatchEmbedding): 图像切块嵌入层。
        cls_token (nn.Parameter): 分类令牌，形状 [1, 1, d_model]。
        pos_embed (nn.Parameter): 位置编码，形状 [1, n_patches+1, d_model]。
        encoder (Encoder): Transformer 编码器。
        norm (nn.LayerNorm): 最终输出的层归一化。
        head (nn.Linear): 分类头。
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 n_classes: int = 1000, d_model: int = 768, n_layers: int = 12,
                 n_head: int = 12, d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()

        # 1. 图像切块与嵌入层
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, d_model)
        num_patches = self.patch_embed.n_patches

        # 2. 定义 [CLS] Token
        # 这是一个可学习的向量，用于聚合整个图像的表征
        # 形状: [1, 1, d_model] -> Batch 维度为 1 以便广播
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3. 定义位置编码 (Position Embedding)
        # 也是可学习的参数。长度 = 图像块数量 + 1个 CLS Token
        # 形状: [1, n_patches + 1, d_model]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        # 位置编码后的 Dropout
        self.pos_drop = nn.Dropout(p=dropout)

        # 4. Transformer Encoder
        # 注意：虽然 Encoder 初始化需要 vocab_size，但在 ViT 模式下我们不使用它的 Embedding 层。
        # 这里传入 1 仅作为占位符。
        self.encoder = Encoder(
            d_model=d_model,
            n_vocab=1,  # Dummy value
            max_len=num_patches + 1,
            n_layers=n_layers,
            n_head=n_head,
            d_ff=d_ff,
            dropout=dropout
        )

        # 5. LayerNorm (Pre-Logits Norm)
        self.norm = nn.LayerNorm(d_model)

        # 6. 分类头 (MLP Head)
        self.head = nn.Linear(d_model, n_classes)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """对关键参数进行截断正态分布初始化，有助于 ViT 收敛。"""
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

        # 初始化 PatchEmbedding 中的卷积层 (如果是 Conv2d 实现)
        if isinstance(self.patch_embed.proj, nn.Conv2d):
            nn.init.kaiming_normal_(self.patch_embed.proj.weight, mode='fan_out')
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)

        # 初始化分类头
        self.head.weight.data.mul_(0.001)
        self.head.bias.data.mul_(0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑。

        Args:
            x (torch.Tensor): 输入图像张量。
                Shape: [Batch, Channel, Height, Width]

        Returns:
            torch.Tensor: 分类 logits。
                Shape: [Batch, n_classes]
        """
        # x: [B, C, H, W]

        # 1. 获取 Patch Embedding -> [B, N, D]
        x = self.patch_embed(x)

        # 2. 拼接 [CLS] Token
        # 使用 .expand() 将 [1, 1, D] 扩展为 [B, 1, D]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 拼接到序列头部 -> [B, N+1, D]
        x = torch.cat((cls_token, x), dim=1)

        # 3. 加上位置编码
        # 利用广播机制: [B, N+1, D] + [1, N+1, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. 传入 Encoder
        # [关键]: 使用 vit_input 参数显式传入向量，跳过 Encoder 内部的 Embedding 层
        # mask 为 None，因为 ViT 也是全注意力机制
        x = self.encoder(src=None, vit_input=x, mask=None)

        # 5. 取出 [CLS] Token 的输出
        # 对输出进行归一化，然后只取第 0 个 Token (CLS)
        x = self.norm(x)
        cls_output = x[:, 0]  # Shape: [B, D]

        # 6. 分类
        x = self.head(cls_output)  # Shape: [B, n_classes]

        return x