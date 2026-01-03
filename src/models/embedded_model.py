"""
CNN与Transformer嵌入式融合模型实现
嵌入式架构：在CNN主干中集成Transformer注意力模块，类似MobileViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """标准卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D] - 输入序列
        Returns:
            output: [B, N, D] - 输出序列
        """
        B, N, D = x.shape
        
        # 线性投影
        Q = self.W_q(x).view(B, N, self.n_head, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.W_k(x).view(B, N, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_head, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [B, H, N, d_k]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        
        # 输出投影
        output = self.W_o(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer块（用于嵌入CNN中）"""
    
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D] - 输入序列
        Returns:
            output: [B, N, D] - 输出序列
        """
        # 自注意力 + 残差连接
        attn_output = self.attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # FFN + 残差连接
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT风格的嵌入块
    结合局部卷积和全局Transformer注意力
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        d_model=256,
        n_head=4,
        n_layers=2,
        d_ff=512,
        patch_size=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 局部特征提取（3x3卷积）
        self.local_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, d_model, kernel_size=1, stride=1, padding=0)
        )
        
        # Transformer层（全局建模）
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 投影回卷积特征空间
        self.proj_conv = nn.Sequential(
            ConvBlock(d_model, out_channels, kernel_size=1, stride=1, padding=0)
        )
        
        # Fusion融合（残差连接）
        if in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - 输入特征图
        Returns:
            output: [B, C', H, W] - 输出特征图
        """
        B, C, H, W = x.shape
        
        # 保存原始输入用于残差连接
        identity = self.shortcut(x)
        
        # 1. 局部卷积
        x = self.local_conv(x)  # [B, d_model, H, W]
        
        # 2. 展开为patch序列
        # [B, d_model, H, W] -> [B, d_model, H*W] -> [B, H*W, d_model]
        x = x.flatten(2).transpose(1, 2)
        
        # 3. Transformer全局建模
        x = self.transformer(x)  # [B, H*W, d_model]
        
        # 4. 重塑回特征图
        # [B, H*W, d_model] -> [B, d_model, H*W] -> [B, d_model, H, W]
        x = x.transpose(1, 2).view(B, self.d_model, H, W)
        
        # 5. 投影回输出通道
        x = self.proj_conv(x)  # [B, out_channels, H, W]
        
        # 6. 残差连接
        x = x + identity
        
        return x


class EmbeddedModel(nn.Module):
    """
    嵌入式融合模型
    在CNN主干网络中嵌入Transformer注意力模块
    
    架构流程：
    1. Stem: 初始卷积层
    2. Stage 1: 标准卷积块
    3. Stage 2: 标准卷积块
    4. Stage 3: MobileViT块（嵌入Transformer）
    5. Stage 4: MobileViT块（嵌入Transformer）
    6. Classification Head: 全局池化+分类
    
    Args:
        num_classes: 分类类别数
        in_channels: 输入图像通道数
        base_channels: 基础通道数
        transformer_channels: Transformer的隐藏维度
        n_head: Transformer的注意力头数
        n_transformer_layers: 每个MobileViT块的Transformer层数
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        num_classes=4,
        in_channels=3,
        base_channels=32,
        transformer_channels=256,
        n_head=4,
        n_transformer_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        # Stem (初始卷积)
        self.stem = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            ConvBlock(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        )
        
        # Stage 1: 纯卷积
        self.stage1 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            ConvBlock(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1)
        )
        
        # Stage 2: 纯卷积
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            ConvBlock(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
            ConvBlock(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1)
        )
        
        # Stage 3: MobileViT块（嵌入Transformer）
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 6, kernel_size=3, stride=2, padding=1),
            MobileViTBlock(
                in_channels=base_channels * 6,
                out_channels=base_channels * 6,
                d_model=transformer_channels,
                n_head=n_head,
                n_layers=n_transformer_layers,
                d_ff=transformer_channels * 2,
                dropout=dropout
            )
        )
        
        # Stage 4: MobileViT块（嵌入Transformer）
        self.stage4 = nn.Sequential(
            ConvBlock(base_channels * 6, base_channels * 8, kernel_size=3, stride=2, padding=1),
            MobileViTBlock(
                in_channels=base_channels * 8,
                out_channels=base_channels * 8,
                d_model=transformer_channels,
                n_head=n_head,
                n_layers=n_transformer_layers,
                d_ff=transformer_channels * 2,
                dropout=dropout
            )
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, H, W] - 输入图像 (例如: [B, 3, 224, 224])
        
        Returns:
            logits: [B, num_classes] - 分类输出
        """
        # Stem
        x = self.stem(x)  # [B, 32, 112, 112]
        
        # Stage 1 (纯卷积)
        x = self.stage1(x)  # [B, 64, 56, 56]
        
        # Stage 2 (纯卷积)
        x = self.stage2(x)  # [B, 128, 28, 28]
        
        # Stage 3 (嵌入Transformer)
        x = self.stage3(x)  # [B, 192, 14, 14]
        
        # Stage 4 (嵌入Transformer)
        x = self.stage4(x)  # [B, 256, 7, 7]
        
        # 全局平均池化
        x = self.avgpool(x)  # [B, 256, 1, 1]
        x = torch.flatten(x, 1)  # [B, 256]
        
        # 分类
        logits = self.classifier(x)  # [B, num_classes]
        
        return logits


if __name__ == "__main__":
    # 测试嵌入式融合模型
    model = EmbeddedModel(num_classes=4)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"嵌入式融合模型总参数量: {total_params / 1e6:.2f}M")
    
    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 嵌入式融合模型测试通过")
