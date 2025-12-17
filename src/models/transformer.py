"""
src.models.transformer
~~~~~~~~~~~~~~~~~~~~~~

Transformer 模型总装文件。
"""

import torch
import torch.nn as nn

from .blocks.encoder import Encoder
from .blocks.decoder import Decoder


class Transformer(nn.Module):
    """
    Transformer 标准架构。

    架构: [Encoder] -> [Decoder] -> [Linear Projection] -> [Softmax]
    """

    def __init__(self, src_vocab_size, trg_vocab_size, src_max_len, trg_max_len,
                 d_model=512, n_layers=6, n_head=8, d_ff=2048, dropout=0.1, padding_idx=0):
        super().__init__()

        self.padding_idx = padding_idx

        # 初始化 Encoder 和 Decoder
        self.encoder = Encoder(d_model, src_vocab_size, src_max_len, n_layers, n_head, d_ff, dropout, padding_idx)
        self.decoder = Decoder(d_model, trg_vocab_size, trg_max_len, n_layers, n_head, d_ff, dropout, padding_idx)

        # 输出投影层: d_model -> trg_vocab_size
        self.output_linear = nn.Linear(d_model, trg_vocab_size)

    def make_src_mask(self, src: torch.Tensor):
        """
        生成源语言掩码: 仅仅屏蔽 Padding。
        Shape: [batch, src_len] -> [batch, 1, 1, src_len]
        """

        mask = (src != self.padding_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def make_trg_mask(self, trg: torch.Tensor):
        """
        生成目标语言掩码: Padding Mask + Subsequent Mask (上三角)。
        """
        batch_size, trg_len = trg.shape

        # 1. Padding Mask (屏蔽 0)
        # Shape: [batch, 1, 1, trg_len]
        trg_pad_mask = (trg != self.padding_idx).unsqueeze(1).unsqueeze(2)

        # 2. Subsequent Mask (屏蔽未来)
        # 关键点：必须生成一个 (trg_len, trg_len) 的方阵
        # torch.tril: 生成下三角矩阵 (Lower Triangle) -> 1 代表可见，0 代表未来(不可见)
        # .bool(): 转为布尔型，方便后续做 & 运算
        # .to(trg.device): 这一点至关重要！新建的 tensor 默认在 cpu，必须搬到和 trg 一样的设备上
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()

        # 3. 组合掩码 (逻辑与)
        # 广播机制: [batch, 1, 1, trg_len] & [trg_len, trg_len]
        # 最终广播为: [batch, 1, trg_len, trg_len]
        mask = trg_pad_mask & trg_sub_mask

        return mask

    def forward(self, src, trg):
        """
        前向传播。
        Args:
            src: [batch, src_len]
            trg: [batch, trg_len]
        Returns:
            output: [batch, trg_len, trg_vocab_size] (未做 softmax 的 logits)
        """

        # 1. 生成掩码
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # 2. 编码器 (Encoder)
        # encoder_output: [batch, src_len, d_model]
        encoder_output = self.encoder(src, src_mask)

        # 3. 解码器 (Decoder)
        # decoder_output: [batch, trg_len, d_model]
        decoder_output = self.decoder(trg, encoder_output, src_mask, trg_mask)

        # 4. 输出层 (Linear Project)
        # output: [batch, trg_len, trg_vocab_size]
        output = self.output_linear(decoder_output)

        return output