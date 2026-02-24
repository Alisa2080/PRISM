import torch

import warnings
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from torch.nn.init import xavier_normal_,xavier_uniform_,constant_
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Optional
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear as _LinearWithBias
from torch.overrides import has_torch_function, handle_torch_function

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(nn.Linear(dim1, dim2), nn.GELU(), nn.AlphaDropout(p=dropout, inplace=False))

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported activation function: {hidden_act}. Supported: {list(ACT2FN.keys())}")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# class RoutingNetwork(nn.Module):
#     """compute the routing logits for each expert"""
#     def __init__(self, num_experts, dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(dim * 2, dim),
#             RMSNorm(dim),
#             nn.GELU(),
#             nn.Linear(dim, num_experts)
#         )

#     def forward(self, x1, x2):
#         # avg pooling for x1 and x2
#         x1_mean = x1.mean(dim=1)  # [b, dim]
#         x2_mean = x2.mean(dim=1)  # [b, dim]
#         x = torch.cat([x1_mean, x2_mean], dim=-1)  # [b, dim * 2]
#         logits = self.fc(x)  # [b, num_experts]
#         return logits  # return logits, distributed in MoELayer

class RoutingNetwork(nn.Module):
    """compute the routing logits for each expert"""
    def __init__(self, num_experts, dim):
        super().__init__()
        # 使用注意力池化替代均值池化
        self.pool_x1 = SelfAttentionPooling(input_dim=dim)
        self.pool_x2 = SelfAttentionPooling(input_dim=dim)
        
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            RMSNorm(dim),
            nn.GELU(),
            nn.Linear(dim, num_experts)
        )

    def forward(self, x1, x2):
        
        # 使用注意力池化替代均值池化
        x1_pooled = self.pool_x1(x1)  # [b, dim]
        x2_pooled = self.pool_x2(x2)  # [b, dim]
        
        x = torch.cat([x1_pooled, x2_pooled], dim=-1)  # [b, dim * 2]
        logits = self.fc(x)  # [b, num_experts]
        return logits


class LinearFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.AlphaDropout(p=0.25, inplace=False)) #nn.Linear(dim, dim)

    
    def forward(self, x1, x2):
        b, n1, dim = x1.size() # x1:[1, 266, 256] x2:[1, 64, 256]
        _, n2, _ = x2.size()
        combined = torch.cat([x1, x2], dim=1)  # [b, n1+n2, dim] [1, 330, 256]
        out = self.linear(combined)
        #print("---linear img",out[:, n1:, :].size())
        return out[:, :n1, :], out[:, n1:, :] # [b, n1, dim] [1, 266, 256], [b, n2, dim] [1, 64, 256]


class AddFusion(nn.Module):
     def __init__(self, dim, num_pathway):
        super().__init__()
        # 针对不同维度对齐情况的线性层
        self.pathway_to_roi = nn.Sequential(
            nn.Linear(num_pathway, 512), nn.GELU(),
            nn.Linear(512, dim), nn.AlphaDropout(p=0.25)
        )
        self.roi_transform = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.AlphaDropout(p=0.25)
        )
        self.roi_to_pathway = nn.Sequential(
            nn.Linear(dim, 512), nn.GELU(),
            nn.Linear(512, num_pathway), nn.AlphaDropout(p=0.25)
        )
        
     def forward(self, x1, x2):
        """
        x1: pathomics [b, n_roi, dim] 
        x2: genomics [b, n_pathway, dim]
        假设 n_roi > n_pathway (在Ciallo中都被压缩为num_pathway了)
        """
        _, n1, dim = x1.size()
        _, n2, _ = x2.size()
        
        if n1 == n2:
            # 如果维度相同，直接加权融合
            x1_proj = self.roi_transform(x1)
            x2_proj = self.roi_transform(x2)
            fused = x1_proj + x2_proj
            return fused, fused
        elif n1 < n2:
            # x1序列更短，将其投影到x2的长度
            x1_aligned = self.pathway_to_roi(x1.transpose(-2, -1)).transpose(-2, -1)
            x2_proj = self.roi_transform(x2)
            fused = x1_aligned + x2_proj
            return fused[:, :n1, :], fused
        else:
            # x2序列更短，将其投影到x1的长度  
            x1_proj = self.roi_transform(x1)
            x2_aligned = self.roi_to_pathway(x2.transpose(-2, -1)).transpose(-2, -1)
            fused = x1_proj + x2_aligned
            return fused, fused[:, :n2, :]


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8)
    
    def forward(self, x1, x2):
        # x1, x2: [b, n, dim]
        b, n1, dim = x1.size()
        _, n2, _ = x2.size()
        x1_permuted = x1.permute(1, 0, 2)  # [n1, b, dim]
        x2_permuted = x2.permute(1, 0, 2)  # [n2, b, dim]
        combined = torch.cat([x1_permuted, x2_permuted], dim=0)  # [n1 + n2, b, dim]
        attn_output, _ = self.attn(combined, combined, combined)  # [n1 + n2, b, dim]
        x1_attn = attn_output[:n1, :, :].permute(1, 0, 2)  # [b, n1, dim]
        x2_attn = attn_output[n1:, :, :].permute(1, 0, 2)  # [b, n2, dim]
        #print("-----x1_att",x1_attn.size())
        #print("-----x2_att",x2_attn.size())

        return x1_attn, x2_attn

class IdentityFusion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        return x1, x2 

class MoE(nn.Module):
    def __init__(self, num_pathway, dim=256):
        super().__init__()
        self.LinearFusion = LinearFusion(dim)
        self.AddFusion = AddFusion(dim, num_pathway)
        self.AttentionFusion = AttentionFusion(dim)
        self.IdentityFusion = IdentityFusion()
        self.num_experts = 4 
        self.routing_network = RoutingNetwork(self.num_experts, dim=dim)
        self.pre_norm_x1 = RMSNorm(dim)
        self.pre_norm_x2 = RMSNorm(dim)
        self.expert_list = nn.ModuleList([
            self.LinearFusion,
            self.AddFusion,
            self.AttentionFusion,
            self.IdentityFusion
        ])

    def forward(self, x1, x2, k=None):
        """
        x1, x2: two modalities, shape [b, n, dim]
        k: num of expert, default: None(soft mode)
        """
        # 对输入进行前置归一化
        x1_norm = self.pre_norm_x1(x1)  # [b, n, dim]
        x2_norm = self.pre_norm_x2(x2)  # [b, n, dim]
        
        # 使用归一化后的特征进行路由
        logits = self.routing_network(x1_norm, x2_norm)  # [b, num_experts]
        bsz = x1.size(0)
        num_experts = self.num_experts
        router_weights = torch.softmax(logits, dim=-1)

        if k is None or k >= num_experts:
            # soft mode, weighted sum for all experts 
            weights = router_weights  # [b, num_experts]
            out_gene = []
            out_img = []
            for expert in self.expert_list:
                # 使用归一化后的特征进行专家计算
                out1, out2 = expert(x1_norm, x2_norm)  # [b, n, dim]
                out_gene.append(out1.unsqueeze(1))  # [b, 1, n, dim]
                out_img.append(out2.unsqueeze(1))   # [b, 1, n, dim]
            # gather output of all experts
            gene = torch.cat(out_gene, dim=1)  # [b, num_experts, n, dim]
            img = torch.cat(out_img, dim=1)    # [b, num_experts, n, dim]
            # weighted sum
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # [b, num_experts, 1, 1]
            gene = (gene * weights).sum(dim=1)  # [b, n, dim]
            img = (img * weights).sum(dim=1)    # [b, n, dim]

        elif k == 1:
            # hard mode, choose the one with the largest weight
            max_idx = logits.argmax(dim=-1)  # [b]
            out_gene = []
            out_img = []
            for i in range(bsz):
                idx = max_idx[i].item()
                expert = self.expert_list[idx]
                # 使用归一化后的特征进行专家计算
                out1, out2 = expert(x1_norm[i:i+1], x2_norm[i:i+1])  # [1, n, dim]
                out_gene.append(out1)
                out_img.append(out2)
            gene = torch.cat(out_gene, dim=0)  # [b, n, dim]
            img = torch.cat(out_img, dim=0)    # [b, n, dim]

        else:
            # top-k mode: choose the top k weight for k experts
            topk_logits, topk_indices = logits.topk(k, dim=-1)  # [b, k]
            # softmax for top k
            topk_weights = torch.softmax(topk_logits, dim=-1)  # [b, k]
            out_gene = []
            out_img = []
            for expert in self.expert_list:
                # 使用归一化后的特征进行专家计算
                out1, out2 = expert(x1_norm, x2_norm)  # [b, n, dim]
                out_gene.append(out1.unsqueeze(1))  # [b, 1, n, dim]
                out_img.append(out2.unsqueeze(1))   # [b, 1, n, dim]
            # gather output for top k experts
            gene_all = torch.cat(out_gene, dim=1)  # [b, num_experts, n, dim]
            img_all = torch.cat(out_img, dim=1)    # [b, num_experts, n, dim]
            gene = []
            img = []
            for i in range(bsz):
                indices = topk_indices[i]  # [k]
                weights_i = topk_weights[i]  # [k]
                weights_i = weights_i.unsqueeze(-1).unsqueeze(-1)  # [k, 1, 1]
                gene_i = gene_all[i][indices]  # [k, n, dim]
                img_i = img_all[i][indices]    # [k, n, dim]
                gene_i = (gene_i * weights_i).sum(dim=0)  # [n, dim]
                img_i = (img_i * weights_i).sum(dim=0)    # [n, dim]
                gene.append(gene_i.unsqueeze(0))  # [1, n, dim]
                img.append(img_i.unsqueeze(0))    # [1, n, dim]
            gene = torch.cat(gene, dim=0)  # [b, n, dim]
            img = torch.cat(img, dim=0)    # [b, n, dim]

        return gene, img, router_weights



class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim=256):
        super(SelfAttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, input_dim))
        self.fc = nn.Linear(input_dim, input_dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x_proj = self.fc(x)  # [batch_size, seq_length, input_dim]
        attn_scores = torch.matmul(x_proj, self.query.transpose(-2, -1))  # [batch_size, seq_length, 1]
        attn_scores = attn_scores.squeeze(-1)  # [batch_size, seq_length]
        attn_weights = F.softmax(attn_scores, dim=-1) 
        
        x_pool = torch.matmul(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch_size, input_dim]

        return x_pool

class ClusteringLayer(nn.Module):
    def __init__(self, num_features, num_clusters):
        super(ClusteringLayer, self).__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, num_features))

    def forward(self, x):
        """
        Forward pass of the clustering layer.
        
        Args:
        x : torch.Tensor
            Input tensor of shape (1, n, num_features)
        
        Returns:
        torch.Tensor
            Output tensor of shape (1, num_clusters, num_features)
        """
        # Ensure the input is correctly shaped
        assert x.shape[2] == self.num_features, f"Expected feature dim {self.num_features}, got {x.shape[2]}"
        if x.shape[1] == 0:
            raise RuntimeError("ClusteringLayer: received zero ROI tokens (N=0).")


        
        # Calculate the distance from each input feature vector to each cluster center
        # x shape: (1, n, num_features)
        # cluster_centers shape: (num_clusters, num_features)
        # Expanded x shape: (1, n, 1, num_features)
        # Expanded cluster_centers shape: (1, 1, num_clusters, num_features)
        x_expanded = x.unsqueeze(2)
        centers_expanded = self.cluster_centers.unsqueeze(0).unsqueeze(0)
        
        # Compute distances
        distances = torch.norm(x_expanded - centers_expanded, dim=3)  # shape: (1, n, num_clusters)
        
        # Find the closest input features to each cluster center
        # We use argmin to find the index of the minimum distance
        _, indices = torch.min(distances, dim=1)  # Closest input feature index for each cluster
        
        # Gather the closest features
        selected_features = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, self.num_features))
        
        return selected_features

class SoftCluster(nn.Module):
    """
    Soft assignment clustering layer.
    Given a sequence of tokens x: [batch, num_tokens, dim], learns K cluster centers
    and returns K prototypes via soft assignment aggregation.

    Shapes:
      - input:  (B, N, D)
      - output: (B, K, D)
    """
    def __init__(self, num_features: int, num_clusters: int, tau: float = 1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.tau = nn.Parameter(torch.tensor(tau))
        self.cluster_centers = nn.Parameter(torch.empty(num_clusters, num_features))
        nn.init.xavier_normal_(self.cluster_centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        assert x.dim() == 3 and x.size(-1) == self.num_features
        # squared euclidean distances: [B, N, K]
        x = F.normalize(x, dim=-1)
        centers = F.normalize(self.cluster_centers, dim=-1)
        similarities = torch.einsum('bnd,kd->bnk', x, centers)
        # soft assignment over tokens for each cluster: sum over N = 1
        weights = torch.softmax(similarities / self.tau.clamp(min=1e-6), dim=1)
        # aggregate tokens to prototypes per cluster
        prototypes = torch.einsum('bnk,bnd->bkd', weights, x)  # [B, K, D]
        return prototypes

    
class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """

    def __init__(
        self,
        skip=0,               # 是否在最终输出中添加原始特征（跳跃连接）
        use_bilinear=0,       # 是否在门控机制中使用双线性池化
        gate1=1,              # 是否对模态1应用门控机制
        gate2=1,              # 是否对模态2应用门控机制
        dim1=128,             # 模态1的特征维度
        dim2=128,             # 模态2的特征维度
        scale_dim1=1,         # 模态1在线性层前的缩放因子
        scale_dim2=1,         # 模态2在线性层前的缩放因子
        mmhid=256,            # 多模态融合后的特征维度
        dropout_rate=0.25,    # Dropout率
    ):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1 // scale_dim1, dim2 // scale_dim2
        skip_dim = dim1_og + dim2_og if skip else 0

        # 模态1的门控单元
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.GELU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear 
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.GELU(), nn.Dropout(p=dropout_rate))

        # 模态2的门控单元
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.GELU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear 
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.GELU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1), 256), nn.GELU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256 + skip_dim, mmhid), nn.GELU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=o1.device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=o2.device)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)  # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out

class AttentionGatedFusion(nn.Module):
    """
    注意力门控融合：对两模态向量做投影 -> 生成门控 -> 加权融合 -> 输出映射
    in_dim:  两模态输入维度（等于 Ciallo 中 self.dim）
    out_dim: 融合后输出维度（等于 hidden[-1]，和分类器输入一致）
    gate_type: 'scalar' | 'vector'
    """
    def __init__(self, in_dim, out_dim, gate_type="scalar", dropout=0.25):
        super().__init__()
        assert gate_type in ["scalar", "vector"]
        self.gate_type = gate_type

        # 预归一化 + 投影到统一 out_dim
        self.norm_p = RMSNorm(in_dim)
        self.norm_g = RMSNorm(in_dim)
        self.proj_p = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())
        self.proj_g = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())

        # 门控头：基于拼接上下文生成门控权重
        if gate_type == "scalar":
            # 生成两个标量权重 -> softmax -> [α_p, α_g]
            self.gate = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim), nn.GELU(),
                nn.Linear(out_dim, 2)
            )
        else:  # 'vector'，逐维门控
            self.gate = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim), nn.GELU(),
                nn.Linear(out_dim, out_dim * 2)
            )

        # 融合后的小头
        self.post = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, path_vec, gene_vec):
        # 输入: [B, in_dim]
        p = self.proj_p(self.norm_p(path_vec))   # [B, out_dim]
        g = self.proj_g(self.norm_g(gene_vec))   # [B, out_dim]
        ctx = torch.cat([p, g], dim=1)           # [B, 2*out_dim]

        if self.gate_type == "scalar":
            logits = self.gate(ctx)              # [B, 2]
            alpha = torch.softmax(logits, dim=1) # [B, 2]
            p_w = alpha[:, :1]                   # [B, 1]
            g_w = alpha[:, 1:]                   # [B, 1]
            fused = p_w * p + g_w * g            # [B, out_dim]
        else:
            gate = torch.sigmoid(self.gate(ctx)) # [B, 2*out_dim]
            gp, gg = gate.chunk(2, dim=1)        # [B, out_dim] x2
            fused = gp * p + gg * g

        return self.post(fused)                  # [B, out_dim]

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)], dim=1)
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # RMSNorm (parameter-free) on Q and K before computing attention
    # Normalize per head on the last dimension to stabilize attention logits
    rms_eps = 1e-6
    q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
    k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + rms_eps)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) * scaling
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_raw = attn_output_weights
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        if need_raw:
            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights_raw

            # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            # return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_raw, attn_output_weights_raw.sum(dim=1) / num_heads
        else:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, need_raw=True, attn_mask=None):
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
            )
