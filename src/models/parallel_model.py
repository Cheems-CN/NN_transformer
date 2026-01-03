"""
CNN+Transformer并行融合模型实现
并行架构：CNN和Transformer分别独立处理输入，然后融合特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.encoder import Encoder
from src.models.layers.patch_embedding import PatchEmbedding


class CNNBranch(nn.Module):
    """
    CNN分支
    基于ResNet架构提取局部纹理特征
    """
    
    def __init__(self, in_channels=3, base_channels=64, output_dim=512):
        super().__init__()
        
        self.output_dim = output_dim
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet-style blocks
        self.conv2 = self._make_layer(base_channels, base_channels * 2, 2, stride=1)
        self.conv3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.conv4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 映射到统一维度
        self.fc = nn.Linear(base_channels * 8, output_dim)
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """构建ResNet-style层"""
        layers = []
        
        # 第一个block可能有降采样
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ))
        
        # 残差连接
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ))
        
        # 其余blocks
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ))
        
        return nn.ModuleList(layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - 输入图像
        Returns:
            features: [B, output_dim] - CNN特征向量
        """
        x = self.conv1(x)
        
        # Conv2
        identity = x
        for i, block in enumerate(self.conv2):
            if i == 0:
                x = F.relu(block(x))
            elif i == 1:
                identity = block(identity)
                x = F.relu(x + identity)
            else:
                identity = x
                x = F.relu(block(x) + identity)
        
        # Conv3
        identity = x
        for i, block in enumerate(self.conv3):
            if i == 0:
                x = F.relu(block(x))
            elif i == 1:
                identity = block(identity)
                x = F.relu(x + identity)
            else:
                identity = x
                x = F.relu(block(x) + identity)
        
        # Conv4
        identity = x
        for i, block in enumerate(self.conv4):
            if i == 0:
                x = F.relu(block(x))
            elif i == 1:
                identity = block(identity)
                x = F.relu(x + identity)
            else:
                identity = x
                x = F.relu(block(x) + identity)
        
        # 全局池化和降维
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ViTBranch(nn.Module):
    """
    ViT分支
    基于Vision Transformer提取全局特征
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        d_model=512,
        n_layers=6,
        n_head=8,
        d_ff=2048,
        dropout=0.1,
        output_dim=512
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.n_patches
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        self.encoder = Encoder(
            d_model=d_model,
            n_vocab=1,  # Dummy
            max_len=num_patches + 1,
            n_layers=n_layers,
            n_head=n_head,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 输出归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 映射到统一维度
        if d_model != output_dim:
            self.fc = nn.Linear(d_model, output_dim)
        else:
            self.fc = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        if isinstance(self.patch_embed.proj, nn.Conv2d):
            nn.init.kaiming_normal_(self.patch_embed.proj.weight, mode='fan_out')
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - 输入图像
        Returns:
            features: [B, output_dim] - ViT特征向量
        """
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)
        
        # 添加CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer编码
        x = self.encoder(src=None, vit_input=x, mask=None)
        
        # 提取CLS token
        x = self.norm(x)
        cls_output = x[:, 0]
        
        # 映射到统一维度
        features = self.fc(cls_output)
        
        return features


class ParallelModel(nn.Module):
    """
    CNN + Transformer 并行融合模型
    
    架构流程：
    1. CNN分支：提取局部纹理特征
    2. ViT分支：提取全局结构特征
    3. 特征融合：将两个分支的特征向量拼接
    4. 分类头：基于融合特征进行分类
    
    Args:
        num_classes: 分类类别数
        in_channels: 输入图像通道数
        feature_dim: 每个分支的输出特征维度
        img_size: 输入图像尺寸
        patch_size: ViT的patch大小
        vit_d_model: ViT的隐藏层维度
        vit_n_layers: ViT的层数
        vit_n_head: ViT的注意力头数
        vit_d_ff: ViT的FFN维度
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        num_classes=4,
        in_channels=3,
        feature_dim=512,
        img_size=224,
        patch_size=16,
        vit_d_model=512,
        vit_n_layers=6,
        vit_n_head=8,
        vit_d_ff=2048,
        dropout=0.1
    ):
        super().__init__()
        
        # CNN分支
        self.cnn_branch = CNNBranch(
            in_channels=in_channels,
            base_channels=64,
            output_dim=feature_dim
        )
        
        # ViT分支
        self.vit_branch = ViTBranch(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=vit_d_model,
            n_layers=vit_n_layers,
            n_head=vit_n_head,
            d_ff=vit_d_ff,
            dropout=dropout,
            output_dim=feature_dim
        )
        
        # 特征融合层
        # 拼接后的维度是 feature_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 分类头
        self.classifier = nn.Linear(feature_dim // 2, num_classes)
        
        self._init_classifier()
    
    def _init_classifier(self):
        """初始化分类器权重"""
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, H, W] - 输入图像
        
        Returns:
            logits: [B, num_classes] - 分类输出
        """
        # 并行处理
        cnn_features = self.cnn_branch(x)  # [B, feature_dim]
        vit_features = self.vit_branch(x)  # [B, feature_dim]
        
        # 特征融合（拼接）
        fused_features = torch.cat([cnn_features, vit_features], dim=1)  # [B, feature_dim * 2]
        
        # 融合层处理
        fused_features = self.fusion(fused_features)  # [B, feature_dim // 2]
        
        # 分类
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        return logits


if __name__ == "__main__":
    # 测试并行模型
    model = ParallelModel(num_classes=4, feature_dim=512)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"并行融合模型总参数量: {total_params / 1e6:.2f}M")
    
    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 并行融合模型测试通过")
