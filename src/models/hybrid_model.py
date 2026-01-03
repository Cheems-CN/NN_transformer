"""
CNN+Transformer混合模型实现
串行融合架构：CNN作为特征提取前端 + Transformer作为全局建模后端
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.encoder import Encoder


class CNNFeatureExtractor(nn.Module):
    """
    CNN特征提取器
    负责提取局部纹理特征，输出特征图
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # 渐进式特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - 输入图像
        Returns:
            features: [B, C', H', W'] - 特征图
        """
        x = self.conv1(x)  # [B, 64, 56, 56]
        x = self.conv2(x)  # [B, 128, 56, 56]
        x = self.conv3(x)  # [B, 256, 28, 28]
        x = self.conv4(x)  # [B, 512, 14, 14]
        
        return x


class HybridModel(nn.Module):
    """
    CNN + Transformer 混合模型
    
    架构流程：
    1. CNN Feature Extractor: 提取局部纹理特征
    2. Feature Map to Sequence: 将特征图转换为序列
    3. Transformer Encoder: 全局依赖建模
    4. Classification Head: 分类输出
    
    Args:
        num_classes: 分类类别数
        in_channels: 输入图像通道数
        d_model: Transformer隐藏层维度
        n_layers: Transformer层数
        n_head: 多头注意力头数
        d_ff: FFN中间层维度
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        num_classes=4,
        in_channels=3,
        d_model=512,
        n_layers=6,
        n_head=8,
        d_ff=2048,
        dropout=0.1
    ):
        super().__init__()
        
        # 1. CNN特征提取器
        self.cnn_extractor = CNNFeatureExtractor(in_channels=in_channels, base_channels=64)
        
        # CNN输出通道数（根据CNNFeatureExtractor的设计，最后输出512通道）
        cnn_output_channels = 512
        
        # 2. 将CNN特征映射到Transformer维度
        self.feature_projection = nn.Linear(cnn_output_channels, d_model)
        
        # 3. 可学习的CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 4. 位置编码（可学习）
        # 假设特征图为14x14，序列长度为196+1（加上CLS）
        max_seq_len = 14 * 14 + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 5. Transformer Encoder
        self.encoder = Encoder(
            d_model=d_model,
            n_vocab=1,  # Dummy，不使用embedding
            max_len=max_seq_len,
            n_layers=n_layers,
            n_head=n_head,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 6. 输出归一化和分类头
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, H, W] - 输入图像
        
        Returns:
            logits: [B, num_classes] - 分类输出
        """
        B = x.shape[0]
        
        # 1. CNN特征提取
        # [B, 3, 224, 224] -> [B, 512, 14, 14]
        cnn_features = self.cnn_extractor(x)
        
        # 2. 将特征图展平为序列
        # [B, 512, 14, 14] -> [B, 512, 196] -> [B, 196, 512]
        cnn_features = cnn_features.flatten(2).transpose(1, 2)
        
        # 3. 投影到Transformer维度
        # [B, 196, 512] -> [B, 196, d_model]
        x = self.feature_projection(cnn_features)
        
        # 4. 添加CLS token
        # [1, 1, d_model] -> [B, 1, d_model]
        cls_token = self.cls_token.expand(B, -1, -1)
        # [B, 196, d_model] -> [B, 197, d_model]
        x = torch.cat((cls_token, x), dim=1)
        
        # 5. 添加位置编码
        # 截取实际需要的位置编码长度
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.pos_drop(x)
        
        # 6. Transformer编码
        # 使用vit_input参数传入，跳过embedding层
        x = self.encoder(src=None, vit_input=x, mask=None)
        
        # 7. 提取CLS token并分类
        x = self.norm(x)
        cls_output = x[:, 0]  # [B, d_model]
        logits = self.head(cls_output)  # [B, num_classes]
        
        return logits


if __name__ == "__main__":
    # 测试混合模型
    model = HybridModel(num_classes=4, d_model=512, n_layers=4, n_head=8)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"混合模型总参数量: {total_params / 1e6:.2f}M")
    
    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 混合模型测试通过")
