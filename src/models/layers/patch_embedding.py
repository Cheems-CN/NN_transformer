"""
Vision Transformer (ViT) 的 Patch Embedding 层实现模块。

本模块提供了 `PatchEmbedding` 类，负责将输入的 2D 图像切分为互不重叠的
Patch 序列，并通过线性投影将其映射到潜在的嵌入空间。这是 ViT 架构的数据输入预处理阶段。

参考文献:
    Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021.
    https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    2D 图像 Patch 嵌入层 (Image to Patch Embedding).

    将输入图像切分为互不重叠的 Patch，并将其映射为 Transformer 可处理的 1D 向量序列。
    这相当于 NLP 中的 Word Embedding 层，区别在于这里处理的是图像块而非单词。

    Args:
        img_size (int): 输入图像的高度/宽度 (默认: 224)。
        patch_size (int): 每个 Patch 的高度/宽度 (默认: 16)。
        in_chans (int): 输入图像的通道数 (默认: 3, 即 RGB)。
        d_model (int): 嵌入向量的维度/Transformer 的隐藏层维度 (默认: 768)。

    Attributes:
        img_size (int): 图像的高度/宽度。
        patch_size (int): Patch 的高度/宽度。
        n_patches (int): 生成的 Patch 总数量 (Sequence Length)。计算公式为 (img_size // patch_size) ** 2。
        proj (nn.Conv2d): 用于同时执行 Patch 切分和线性映射的卷积层。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 使用 Conv2d 实现 "切分 + 线性投影"
        # kernel_size=stride=patch_size 保证了 Patch 之间无重叠
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑。

        Args:
            x (torch.Tensor): 输入图像张量, 形状 [Batch, in_chans, Height, Width]

        Returns:
            torch.Tensor: 嵌入后的序列张量, 形状 [Batch, n_patches, d_model]
        """
        # x shape: [B, C, H, W]
        # 1. 卷积投影 (Projection): [B, C, H, W] -> [B, d_model, H/P, W/P]
        x = self.proj(x)

        # 2. 展平 (Flatten): 将高和宽展平为序列长度
        # [B, d_model, H/P, W/P] -> [B, d_model, n_patches]
        x = x.flatten(2)

        # 3. 维度交换 (Transpose): 调整为 Transformer 需要的 [Batch, Seq, Dim]
        # [B, d_model, n_patches] -> [B, n_patches, d_model]
        x = x.transpose(1, 2)

        return x