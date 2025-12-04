import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upsampling then single conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SingleConv(in_channels, out_channels) 
        self.joinconv = SingleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 is not None:
            x1 = self.conv(x1)
            h1, w1 = x1.size()[2], x1.size()[3]
            h2, w2 = x2.size()[2], x2.size()[3]
            diffY = h2 - h1
            diffX = w2 - w1
            # 处理尺寸不匹配: 裁剪x2
            x2 = x2[:, :, diffY // 2: h2 - (diffY - diffY // 2), diffX // 2: w2 - (diffX - diffX // 2)]
            x1 = torch.cat([x2, x1], dim=1)
        return self.joinconv(x1)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim=768, patch_size=1, W=16, H=16):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.pos_embed = nn.Parameter(torch.zeros(1, H * W // (patch_size * patch_size), embed_dim)) 
        nn.init.trunc_normal_(self.pos_embed, std=.02) 
        
    def forward(self, x):
        z = self.proj(x)
        z = z.flatten(2).transpose(1, 2)
        z = z + self.pos_embed
        return z

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=512, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) 
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class SingleConv(nn.Module):
    """(conv => BN => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.single_conv(x)

class TransUnet(nn.Module):
    def __init__(self, 
                 n_channels, 
                 n_classes, 
                 embed_dim=128,
                 img_size=256,
                #  feature_size=64,
                 num_heads=8,
                 num_layers=4,
                #  hidden_dim=256,
                 mlp_dim=512,
                 dropout_rate=0.1,
                #  attention_dropout_rate=0.1
                 ):
        super(TransUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embed_dim = embed_dim
    
        # 编码器
        # 原始UNet编码器
        # self.inc = DoubleConv(n_channels, 64)  # (B, 64, H, W)
        # self.down1 = Down(64, 128)  # (B, 128, H/2, W/2)
        # self.down2 = Down(128, 256)  # (B, 256, H/4, W/4)
        # self.down3 = Down(256, 512)  # (B, 512, H/8, W/8)
        # self.down4 = Down(512, 1024)  # (B, 1024, H/16, W/16)
        self.inc = DoubleConv(n_channels, 32)  # (B, 64, H, W)
        self.down1 = Down(32, 64)  # (B, 128, H/2, W/2)
        self.down2 = Down(64, 128)  # (B, 256, H/4, W/4)
        self.down3 = Down(128, 256)  # (B, 256, H/8, W/8)
        self.down4 = Down(256, 512)  # (B, 512, H/16, W/16)
        # 瓶颈层 - Transformer
        H0, W0 = img_size, img_size  # 假设输入图像大小为256x256，则经过4次下采样后为16x16
        self.patch_embed = PatchEmbedding(in_channels=512, embed_dim=embed_dim, patch_size=1, W=W0 // 16, H=H0 // 16)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim=embed_dim, num_heads=num_heads, 
                             mlp_dim=mlp_dim, 
                             dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.upsample_conv = nn.Conv2d(embed_dim, 512, kernel_size=1)  # (B, 512, H/16, W/16)

        # 解码器
        self.up1 = Up(512, 256) 
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 16)
        self.conv_last = SingleConv(16, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # (B, 256, H/8, W/8)
        x5 = self.down4(x4) # (B, 512, H/16, W/16)
        
        # 瓶颈层 - Transformer
        B, C, H, W = x5.shape
        z = self.patch_embed(x5) 
        for layer in self.transformer_layers:
            z = layer(z)
        
        # Reshape: (B, N, 768) -> (B, 768, H/16, W/16)
        z = z.transpose(1, 2).view(B, self.embed_dim, H, W) 
        
        # Conv: (B, 768, H/16, W/16) -> (B, 512, H/16, W/16)
        x = self.upsample_conv(z) 

        # 解码器路径
        x = self.up1(x, x4)  # (B, 256, H/8, W/8)
        x = self.up2(x, x3)  # (B, 128, H/4, W/4)
        x = self.up3(x, x2)  # (B, 64, H/2, W/2)
        x = self.up4(x, None)  # (B, 16, H, W)
        x = self.conv_last(x)  # (B, 16, H, W)
        logits = self.outc(x)  # (B, n_classes, H, W)
        return logits