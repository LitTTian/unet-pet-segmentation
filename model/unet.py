import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Unet双卷积模块(Double Convolution Module)
    执行两组卷积操作，每组卷积后都有一组批归一化和ReLU激活函数。
    结构: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
    作用: 提取输入特征，增加特征图通道数/调整维度
    备注: 和论文中不同，这里通过设置padding=1确保输入输出尺寸一致
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels  # Unet论文中间通道数等于输出通道数
        self.double_conv = nn.Sequential(
            # 第一组卷积
            # 卷积核3x3，padding=1保持尺寸不变
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # 层归一化，稳定训练过程
            nn.BatchNorm2d(mid_channels),
            # inplace=True表示直接在输入上进行操作，节省内存
            nn.ReLU(inplace=True),
            # 第二组卷积
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Unet下采样模块(Downsampling Module)
    结构: MaxPool2d -> DoubleConv
    作用: 通过最大池化降低特征图尺寸，同时提取更深层次的特征
    备注: 论文中有四次下采样，图片尺寸需为16的倍数保证输出尺寸正确
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """
    Unet上采样模块(Upsampling Module)
    结构: ConvTranspose2d -> 拼接(skip-connection) -> DoubleConv
    作用: 通过转置卷积上采样特征图，结合编码器对应层的特征，融合早期细节特征
    备注: 论文中有四次上采样
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 和论文中一样，使用转置卷积进行上采样
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        h1, w1 = x1.size()[2], x1.size()[3]
        h2, w2 = x2.size()[2], x2.size()[3]
        diffY = h2 - h1
        diffX = w2 - w1
        # 和论文中一致，对x2进行剪裁并且和x1拼接
        x2 = x2[:, :, diffY // 2: h2 - (diffY - diffY // 2), diffX // 2: w2 - (diffX - diffX // 2)]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 1x1卷积，相当于线性分类器
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        # Unet编码器部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Unet解码器部分(skip-connections连接前期特征)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        