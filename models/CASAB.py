import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CosinConv import CosinConv2D

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ConvBlock, self).__init__()
        # 如果没有设置输出通道，就保持和输入通道一致
        if out_channels is None:
            out_channels = in_channels

        # 深度卷积：每个通道单独卷积，提取局部特征
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        # 逐点卷积：用 1x1 卷积整合通道信息
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        # 批归一化：加速收敛，提升训练稳定性
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 激活函数：LeakyReLU，避免神经元完全失效
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # 第一步：先做深度卷积 + BN + 激活
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)

        # 第二步：再做逐点卷积 + BN + 激活
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 平均池化：取每个通道的平均值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化：取每个通道的最大值
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两层全连接（用1x1卷积代替）实现通道权重的学习
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),   # SiLU 激活，平滑版 ReLU
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出范围 [0,1]，作为权重系数
        )

    def forward(self, x):
        # 平均池化结果送入全连接
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化结果送入全连接
        max_out = self.fc(self.max_pool(x))
        # 两个结果相加，形成最终的通道注意力
        scale = avg_out + max_out
        # 输入特征乘以注意力权重
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用可配置的kernel_size，padding自动适配
        padding = kernel_size // 2
        # 先用可配置大小的卷积融合 4 个统计量通道，再用 1x1 卷积压缩
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=padding, groups=1),
            nn.SiLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算每个像素点的通道均值
        mean_out = torch.mean(x, dim=1, keepdim=True)
        # 计算每个像素点的通道最大值
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 计算每个像素点的通道最小值
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        # 计算每个像素点的通道和
        sum_out = torch.sum(x, dim=1, keepdim=True)

        # 把四种统计量拼接在一起，形成 4 通道特征
        pool = torch.cat([mean_out, max_out, min_out, sum_out], dim=1)
        # 卷积得到空间注意力图
        attention = self.conv(pool)
        # 输入特征乘以空间注意力权重
        return x * attention


class CASAB(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel_size=7):
        super(CASAB, self).__init__()
        # 卷积块（深度可分离卷积）
        self.convblock = ConvBlock(in_channels, in_channels)
        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels, reduction)
        # 空间注意力（融合四种统计量），支持可配置的kernel_size
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        # 先提取卷积特征
        x = self.convblock(x)
        # 通道注意力输出
        ca = self.channel_attention(x)
        # 空间注意力输出
        sa = self.spatial_attention(x)
        # 把两个注意力结果相加
        return ca + sa

if __name__ == '__main__':
    # 测试大尺寸输入（原始设置）
    x_large = torch.rand(1, 32, 50, 50)
    model_large = CASAB(in_channels=32, spatial_kernel_size=7)
    y_large = model_large(x_large)
    print(f"大尺寸输入测试:")
    print(f"  输入张量形状: {x_large.shape}")
    print(f"  输出张量形状: {y_large.shape}")
    print(f"  spatial_kernel_size: 7")
    
    # 测试小尺寸输入（适用于MSCFusion的4x4）
    x_small = torch.rand(1, 256, 4, 4)  # 模拟MSCFusion中的特征图
    model_small = CASAB(in_channels=256, spatial_kernel_size=3)
    y_small = model_small(x_small)
    print(f"\n小尺寸输入测试:")
    print(f"  输入张量形状: {x_small.shape}")
    print(f"  输出张量形状: {y_small.shape}")
    print(f"  spatial_kernel_size: 3")
    
    print("\n微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")