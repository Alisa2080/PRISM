import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class EnhancedDepthwiseConvMixer(nn.Module):
    """
    增强版深度卷积混合器，通过多核并行捕捉更丰富的局部模式
    在不显著增加参数的前提下提升特征提取能力
    """
    def __init__(self, dim, kernel_sizes=[3, 5], expansion_factor=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        expanded_dim = int(dim * expansion_factor)
        
        # 多核深度卷积分支：捕捉不同范围的局部依赖
        self.local_convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=k, padding=k//2, groups=dim)
            for k in kernel_sizes
        ])
        
        # 扩张卷积分支：捕捉中程依赖
        dilation = 2
        self.medium_conv = nn.Conv1d(
            dim, dim, kernel_size=3,
            padding=(3-1)*dilation//2, dilation=dilation, groups=dim
        )
        
        # 注意力加权融合不同分支
        num_branches = len(kernel_sizes) + 1  # local_convs + medium_conv
        self.branch_attention = nn.Sequential(
            nn.Linear(dim, num_branches),
            nn.Softmax(dim=-1)
        )
        
        # 点卷积融合
        self.pointwise_expand = nn.Conv1d(dim, expanded_dim, 1)
        self.pointwise_compress = nn.Conv1d(expanded_dim, dim, 1)
        
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        """
        residual = x
        x = self.norm(x)
        
        # 转换为conv1d格式 [batch, dim, seq_len]
        x_conv = x.transpose(1, 2)
        
        # 多分支卷积
        branch_outputs = []
        
        # 局部卷积分支
        for local_conv in self.local_convs:
            branch_outputs.append(local_conv(x_conv))
        
        # 中程卷积分支
        branch_outputs.append(self.medium_conv(x_conv))
        
        # 计算分支注意力权重
        # 使用全局平均池化获得序列级别的表示
        global_feat = x_conv.mean(dim=-1)  # [batch, dim]
        branch_weights = self.branch_attention(global_feat)  # [batch, num_branches]
        
        # 加权融合分支
        weighted_output = torch.zeros_like(branch_outputs[0])
        for i, branch_out in enumerate(branch_outputs):
            weighted_output += branch_weights[:, i:i+1, None] * branch_out
        
        # 点卷积处理
        expanded = self.activation(self.pointwise_expand(weighted_output))
        compressed = self.pointwise_compress(expanded)
        
        # 转换回原格式并加残差
        output = compressed.transpose(1, 2) + residual
        return self.dropout(output)

class DepthwiseConvMixer(nn.Module):
    """
    原始轻量级的深度卷积混合器，用于捕捉局部和中程依赖关系
    比传统自注意力更高效，适合处理序列化的patch特征
    """
    def __init__(self, dim, kernel_size=3, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        expanded_dim = int(dim * expansion_factor)
        
        # 深度卷积分支：捕捉局部依赖
        self.local_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, 
            padding=kernel_size//2, groups=dim
        )
        
        # 扩张卷积分支：捕捉中程依赖  
        dilation = 2
        self.medium_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=(kernel_size-1)*dilation//2, dilation=dilation, groups=dim
        )
        
        # 点卷积融合
        self.pointwise_expand = nn.Conv1d(dim * 2, expanded_dim, 1)
        self.pointwise_compress = nn.Conv1d(expanded_dim, dim, 1)
        
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        """
        residual = x
        x = self.norm(x)
        
        # 转换为conv1d格式 [batch, dim, seq_len]
        x_conv = x.transpose(1, 2)
        
        # 双分支卷积
        local_out = self.local_conv(x_conv)
        medium_out = self.medium_conv(x_conv)
        
        # 融合两个分支
        combined = torch.cat([local_out, medium_out], dim=1)
        expanded = self.activation(self.pointwise_expand(combined))
        compressed = self.pointwise_compress(expanded)
        
        # 转换回原格式并加残差
        output = compressed.transpose(1, 2) + residual
        return self.dropout(output)

class SelfAttentionPooling(nn.Module):
    """
    自注意力池化层，将序列压缩为单一向量表示
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim))
        self.key_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.key_proj.weight)
        nn.init.normal_(self.query, std=0.02)
    
    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        return: [batch, dim]
        """
        keys = self.key_proj(x)  # [batch, seq_len, dim]
        
        # 计算注意力分数
        scores = torch.matmul(keys, self.query.transpose(-2, -1))  # [batch, seq_len, 1]
        weights = F.softmax(scores.squeeze(-1), dim=-1)  # [batch, seq_len]
        
        # 加权求和
        pooled = torch.matmul(weights.unsqueeze(1), x).squeeze(1)  # [batch, dim]
        return self.dropout(pooled)

class ScaleGate(nn.Module):
    """
    原始尺度门控融合模块，自适应地融合不同尺度的特征
    """
    def __init__(self, dim, num_scales=3, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        
        # 门控网络：学习尺度权重
        self.gate_net = nn.Sequential(
            nn.Linear(dim * num_scales, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 可选的尺度特异性变换
        self.scale_transforms = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, scale_features):
        """
        scale_features: list of [batch, dim], 每个元素代表一个尺度的特征
        return: [batch, dim]
        """
        batch_size = scale_features[0].shape[0]
        
        # 应用尺度特异性变换
        transformed_features = []
        for i, feat in enumerate(scale_features):
            transformed = self.scale_transforms[i](feat)
            transformed_features.append(transformed)
        
        # 计算门控权重
        concat_features = torch.cat(transformed_features, dim=-1)  # [batch, dim*num_scales]
        gate_weights = self.gate_net(concat_features)  # [batch, num_scales]
        
        # 加权融合
        fused = torch.zeros_like(transformed_features[0])
        for i, feat in enumerate(transformed_features):
            fused += gate_weights[:, i:i+1] * feat
            
        return self.dropout(fused)

class EnhancedScaleGate(nn.Module):
    """
    增强版尺度门控融合模块，引入了相互注意力和残差连接
    更好地捕捉尺度间的相互依赖关系
    """
    def __init__(self, dim, num_scales=3, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim
        
        # 尺度特异性变换
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            ) for _ in range(num_scales)
        ])
        
        # 跨尺度注意力机制
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 门控网络：基于变换后的特征学习权重
        self.gate_net = nn.Sequential(
            nn.Linear(dim * num_scales, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 最终的融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, scale_features):
        """
        scale_features: list of [batch, dim], 每个元素代表一个尺度的特征
        return: [batch, dim]
        """
        batch_size = scale_features[0].shape[0]
        
        # 应用尺度特异性变换
        transformed_features = []
        for i, feat in enumerate(scale_features):
            transformed = self.scale_transforms[i](feat)
            transformed_features.append(transformed)
        
        # 堆叠为序列进行跨尺度注意力
        scale_stack = torch.stack(transformed_features, dim=1)  # [batch, num_scales, dim]
        
        # 跨尺度注意力：让不同尺度的特征相互交互
        attn_output, _ = self.cross_scale_attn(
            scale_stack, scale_stack, scale_stack
        )  # [batch, num_scales, dim]
        
        # 残差连接
        enhanced_features = scale_stack + attn_output
        
        # 计算门控权重
        concat_features = enhanced_features.flatten(start_dim=1)  # [batch, dim*num_scales]
        gate_weights = self.gate_net(concat_features)  # [batch, num_scales]
        
        # 加权融合
        fused = torch.sum(
            gate_weights.unsqueeze(-1) * enhanced_features, dim=1
        )  # [batch, dim]
        
        # 最终变换
        output = self.fusion_layer(fused)
        return self.dropout(output)

class LightMSF(nn.Module):
    """
    轻量多尺度融合模块 (Light Multi-Scale Fusion)
    
    替代ROAM的PyramidViT，提供更轻量但有效的多尺度特征融合
    保持与Ciallo现有架构的完全兼容性
    """
    def __init__(self, 
                 num_patches=84, 
                 patch_dim=1024, 
                 dim=256,
                 dropout=0.1,
                 extract_scale="all",  # "x20", "x10", "x5", "all"
                 ape=True,  # 是否使用绝对位置编码
                 share_proj=True,  # 是否共享投影层
                 intra_scale="enhanced_conv_mixer",  # "enhanced_conv_mixer", "conv_mixer" or "tiny_attn"
                 inter_scale="enhanced_gate"):  # "enhanced_gate", "gate" or "tiny_transformer"
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.dim = dim
        self.extract_scale = extract_scale
        self.ape = ape
        self.share_proj = share_proj
        self.intra_scale = intra_scale
        self.inter_scale = inter_scale
        
        # 多尺度分割索引 (基于84维结构)
        self.scale_indices = {
            "x20": (0, 64),    # 前64个patch: 20倍放大
            "x10": (64, 80),   # 中间16个patch: 10倍放大  
            "x5": (80, 84),    # 最后4个patch: 5倍放大
            "all": (0, 84)     # 全部84个patch
        }
        
        # 投影层设置
        if share_proj:
            # 共享投影 + 尺度嵌入
            self.shared_proj = nn.Linear(patch_dim, dim)
            self.scale_embeddings = nn.Parameter(torch.randn(3, dim))  # 3个尺度的嵌入
        else:
            # 分尺度投影
            self.proj_x20 = nn.Linear(patch_dim, dim)
            self.proj_x10 = nn.Linear(patch_dim, dim) 
            self.proj_x5 = nn.Linear(patch_dim, dim)
        
        # 绝对位置编码 (固定2D正弦位置编码)
        if ape:
            self.pos_emb_x20 = self._create_2d_pos_embedding(8, 8, dim)    # 8x8=64
            self.pos_emb_x10 = self._create_2d_pos_embedding(4, 4, dim)    # 4x4=16
            self.pos_emb_x5 = self._create_2d_pos_embedding(2, 2, dim)     # 2x2=4
        
        # 尺度内上下文建模
        if intra_scale == "enhanced_conv_mixer":
            self.intra_x20 = EnhancedDepthwiseConvMixer(dim, kernel_sizes=[3, 5], dropout=dropout)
            self.intra_x10 = EnhancedDepthwiseConvMixer(dim, kernel_sizes=[3, 5], dropout=dropout)
            self.intra_x5 = EnhancedDepthwiseConvMixer(dim, kernel_sizes=[3, 5], dropout=dropout)
        elif intra_scale == "conv_mixer":
            # 保留原始版本作为轻量选项
            self.intra_x20 = DepthwiseConvMixer(dim, dropout=dropout)
            self.intra_x10 = DepthwiseConvMixer(dim, dropout=dropout)
            self.intra_x5 = DepthwiseConvMixer(dim, dropout=dropout)
        elif intra_scale == "tiny_attn":
            # 简化的单头自注意力
            self.intra_x20 = nn.MultiheadAttention(dim, num_heads=1, dropout=dropout, batch_first=True)
            self.intra_x10 = nn.MultiheadAttention(dim, num_heads=1, dropout=dropout, batch_first=True)
            self.intra_x5 = nn.MultiheadAttention(dim, num_heads=1, dropout=dropout, batch_first=True)
        
        # 尺度内聚合
        self.pool_x20 = SelfAttentionPooling(dim, dropout)
        self.pool_x10 = SelfAttentionPooling(dim, dropout)
        self.pool_x5 = SelfAttentionPooling(dim, dropout)
        
        # 尺度间融合
        if inter_scale == "enhanced_gate":
            self.scale_fusion = EnhancedScaleGate(dim, num_scales=3, dropout=dropout)
        elif inter_scale == "gate":
            # 保留原始版本作为轻量选项
            self.scale_fusion = ScaleGate(dim, num_scales=3, dropout=dropout)
        elif inter_scale == "tiny_transformer":
            # 简单的3-token Transformer
            self.scale_fusion = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim, 
                    nhead=4, 
                    dim_feedforward=dim*2,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=1
            )
            self.final_pool = SelfAttentionPooling(dim, dropout)
        
        # 初始化
        self.apply(self._init_weights)
        
        print(f"[LightMSF] 初始化完成:")
        print(f"  - extract_scale: {extract_scale}")
        print(f"  - share_proj: {share_proj}")
        print(f"  - intra_scale: {intra_scale}")
        print(f"  - inter_scale: {inter_scale}")
        print(f"  - ape: {ape}")
        
    def _create_2d_pos_embedding(self, height, width, dim):
        """创建2D正弦位置编码"""
        pos_h = torch.arange(height).float()
        pos_w = torch.arange(width).float()
        
        # 创建2D网格
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        grid = torch.stack([grid_h, grid_w], dim=-1).reshape(-1, 2)  # [h*w, 2]
        
        # 正弦位置编码
        pe = torch.zeros(height * width, dim)
        div_term = torch.exp(torch.arange(0, dim//2) * -(math.log(10000.0) / (dim//2)))
        
        pe[:, 0::4] = torch.sin(grid[:, 0:1] * div_term[::2])
        pe[:, 1::4] = torch.cos(grid[:, 0:1] * div_term[::2])
        pe[:, 2::4] = torch.sin(grid[:, 1:2] * div_term[::2])
        pe[:, 3::4] = torch.cos(grid[:, 1:2] * div_term[::2])
        
        return nn.Parameter(pe, requires_grad=False)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def _process_single_scale(self, x, scale_name, intra_module, pool_module):
        """处理单个尺度的特征"""
        # 投影到目标维度
        if self.share_proj:
            x = self.shared_proj(x)
            # 添加尺度嵌入
            scale_idx = {"x20": 0, "x10": 1, "x5": 2}[scale_name]
            x = x + self.scale_embeddings[scale_idx]
        else:
            proj_layer = getattr(self, f"proj_{scale_name}")
            x = proj_layer(x)
        
        # 添加位置编码
        if self.ape:
            pos_emb = getattr(self, f"pos_emb_{scale_name}")
            x = x + pos_emb
        
        # 尺度内上下文建模
        if self.intra_scale in ["enhanced_conv_mixer", "conv_mixer"]:
            x = intra_module(x)
        elif self.intra_scale == "tiny_attn":
            x_attn, _ = intra_module(x, x, x)
            x = x + x_attn  # 残差连接
        
        # 尺度内聚合
        pooled = pool_module(x)  # [batch, dim]
        
        return pooled
    
    def get_pathomics_features(self, x_path):
        """
        主要接口函数，与ROAM的get_pathomics_features完全兼容
        
        Args:
            x_path: [B, K, 84, patch_dim] 输入的路径特征
            
        Returns:
            features: [(B*K), dim] 提取的特征，与ROAM输出格式完全一致
        """
        B, K, num_patches, patch_dim = x_path.shape
        
        # 验证输入维度
        assert num_patches == self.num_patches, f"Expected {self.num_patches} patches, got {num_patches}"
        assert patch_dim == self.patch_dim, f"Expected {self.patch_dim} patch_dim, got {patch_dim}"
        
        # 重排为 [(B*K), 84, patch_dim]
        x = rearrange(x_path, 'b k n d -> (b k) n d')
        batch_size = x.shape[0]  # B*K
        
        # 根据extract_scale设置处理哪些尺度
        if self.extract_scale == "all":
            scales_to_process = ["x20", "x10", "x5"]
        else:
            scales_to_process = [self.extract_scale]
        
        scale_features = []
        
        # 处理每个尺度
        for scale_name in scales_to_process:
            start_idx, end_idx = self.scale_indices[scale_name]
            x_scale = x[:, start_idx:end_idx, :]  # [batch, scale_patches, patch_dim]
            
            # 获取对应的模块
            intra_module = getattr(self, f"intra_{scale_name}")
            pool_module = getattr(self, f"pool_{scale_name}")
            
            # 处理该尺度
            scale_feat = self._process_single_scale(x_scale, scale_name, intra_module, pool_module)
            scale_features.append(scale_feat)
        
        # 尺度间融合
        if len(scale_features) == 1:
            # 单尺度情况
            final_features = scale_features[0]
        else:
            # 多尺度融合
            if self.inter_scale in ["enhanced_gate", "gate"]:
                final_features = self.scale_fusion(scale_features)
            elif self.inter_scale == "tiny_transformer":
                # 将尺度特征堆叠为序列 [batch, 3, dim]
                scale_stack = torch.stack(scale_features, dim=1)
                fused_stack = self.scale_fusion(scale_stack)  # [batch, 3, dim]
                final_features = self.final_pool(fused_stack)  # [batch, dim]
        
        return final_features  # [(B*K), dim]
    
    def forward(self, x_path):
        """
        前向传播函数，调用get_pathomics_features
        """
        return self.get_pathomics_features(x_path)

# 工厂函数，方便创建不同配置的LightMSF
def create_light_msf(config_name="default", **kwargs):
    """
    创建不同配置的LightMSF模型
    
    Args:
        config_name: 配置名称
            - "default": 默认配置，平衡性能和效率
            - "ultra_light": 超轻量配置
            - "performance": 性能优先配置
        **kwargs: 额外的参数覆盖
    """
    configs = {
        "default": {
            "dim": 256,
            "dropout": 0.1,
            "share_proj": True,
            "intra_scale": "conv_mixer",
            "inter_scale": "gate",
            "ape": True
        },
        "ultra_light": {
            "dim": 128,
            "dropout": 0.05,
            "share_proj": True,
            "intra_scale": "conv_mixer",
            "inter_scale": "gate",
            "ape": False
        },
        "performance": {
            "dim": 256,
            "dropout": 0.15,
            "share_proj": False,
            "intra_scale": "enhanced_conv_mixer",
            "inter_scale": "enhanced_gate",
            "ape": True
        },
        "enhanced_performance": {
            "dim": 512,
            "dropout": 0.2,
            "share_proj": False,
            "intra_scale": "enhanced_conv_mixer",
            "inter_scale": "enhanced_gate",
            "ape": True
        }
    }
    
    config = configs.get(config_name, configs["default"])
    config.update(kwargs)  # 用传入的参数覆盖默认配置
    
    return LightMSF(**config)


if __name__ == "__main__":
    # 测试代码
    print("Testing LightMSF...")
    
    # 创建模型
    model = LightMSF(dim=256, extract_scale="all")
    
    # 测试输入
    B, K = 1, 100  # 1个batch，100个ROI
    x_path = torch.randn(B, K, 84, 1024)
    
    print(f"Input shape: {x_path.shape}")
    
    # 前向传播
    with torch.no_grad():
        features = model.get_pathomics_features(x_path)
        print(f"Output shape: {features.shape}")
        print(f"Expected shape: [{B*K}, {model.dim}]")
        
    # 测试不同尺度
    for scale in ["x20", "x10", "x5", "all"]:
        model_scale = LightMSF(dim=256, extract_scale=scale)
        with torch.no_grad():
            features_scale = model_scale.get_pathomics_features(x_path)
            print(f"Scale {scale}: output shape {features_scale.shape}")
    
    print("✓ All tests passed!")
