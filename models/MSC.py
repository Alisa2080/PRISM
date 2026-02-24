
import torch
import torch.nn as nn
import math
from einops import rearrange

class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        """
        参数:
            dim: 输入特征的维度
            num_heads: 注意力头的数量
            topk: 是否使用top-k注意力机制
            kernel: 不同尺度池化操作的卷积核大小列表
            s: 不同尺度池化操作的步长列表
            pad: 不同尺度池化操作的填充大小列表
            qkv_bias: 是否在QKV投影中使用偏置
            qk_scale: 查询键缩放因子，默认为None时使用头维度的平方根
            attn_drop_ratio: 注意力权重的dropout比率
            proj_drop_ratio: 投影层的dropout比率
            k1: 第一种注意力机制选择的top-k比例参数
            k2: 第二种注意力机制选择的top-k比例参数
        """
        super(MSC, self).__init__()
        self.num_heads = num_heads  # 注意力头数量
        head_dim = dim // num_heads  # 每个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 查询键缩放因子

        # 定义QKV线性投影层
        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # 查询投影
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 键值投影
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力权重dropout
        self.proj = nn.Linear(dim, dim)  # 输出投影
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # 输出dropout
        self.k1 = k1  # 第一种注意力机制的top-k参数
        self.k2 = k2  # 第二种注意力机制的top-k参数

        # 可学习的注意力融合权重
        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)  # 第一种注意力的权重
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)  # 第二种注意力的权重

        # 多尺度平均池化层，用于捕获不同尺度的上下文信息
        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)  # 层归一化


    def forward(self, x, y):
        # 多尺度池化操作
        y1 = self.avgpool1(y)  # 第一尺度池化 3
        y2 = self.avgpool2(y)  # 第二尺度池化 5
        y3 = self.avgpool3(y)  # 第三尺度池化 7
        y = y1 + y2 + y3  # 融合多尺度信息

        # 重塑张量为序列形式并应用层归一化
        y = y.flatten(-2, -1)  # 将高度和宽度维度展平
        y = y.transpose(1, 2)  # 转置维度，使序列维度在前
        y = self.layer_norm(y)  # 应用层归一化

        # 处理键值对
        B, N1, C = y.shape  # 获取上下文特征的批次大小、序列长度和维度
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # 分离键和值

        # 重塑输入特征为序列形式
        x = rearrange(x, 'b c h w -> b (h w) c')  # 将输入特征转换为序列形式
        # 处理查询
        B, N, C = x.shape  # 获取输入特征的批次大小、序列长度和维度
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 生成查询

        # 计算注意力得分 X 与 Y
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算查询和键的点积并缩放

        # 第一种注意力机制：基于top-k选择
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)  # 创建掩码
        index = torch.topk(attn, k=int(N1 / self.k1), dim=-1, largest=True)[1]  # 选择得分最高的前k个位置
        mask1.scatter_(-1, index, 1.)  # 在掩码中标记选中的位置
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))  # 应用掩码
        attn1 = attn1.softmax(dim=-1)  # 计算softmax得到注意力权重
        attn1 = self.attn_drop(attn1)  # 应用dropout
        out1 = (attn1 @ v)  # 计算注意力输出

        # 第二种注意力机制：基于top-k选择（不同的k值）
        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)  # 创建掩码
        index = torch.topk(attn, k=int(N1 / self.k2), dim=-1, largest=True)[1]  # 选择得分最高的前k个位置
        mask2.scatter_(-1, index, 1.)  # 在掩码中标记选中的位置
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))  # 应用掩码
        attn2 = attn2.softmax(dim=-1)  # 计算softmax得到注意力权重
        attn2 = self.attn_drop(attn2)  # 应用dropout
        out2 = (attn2 @ v)  # 计算注意力输出

        # 融合两种注意力机制的输出
        out = out1 * self.attn1 + out2 * self.attn2  # 使用可学习的权重融合两种注意力输出

        # 重塑输出张量回原始形状
        x = out.transpose(1, 2).reshape(B, N, C)  # 调整维度顺序并重塑
        x = self.proj(x)  # 应用输出投影
        x = self.proj_drop(x)  # 应用dropout

        # 将序列形式的输出重塑回图像形式
        hw = int(math.sqrt(N))  # 计算原始图像的高度和宽度
        x = rearrange(x, 'b (h w) c -> b c h w', h=hw, w=hw)  # 重塑为图像形式

        return x