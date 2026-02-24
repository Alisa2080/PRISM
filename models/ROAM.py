import torch
from torch import nn
import torch.nn.functional as F
from torch import einsum 
from einops import rearrange, repeat
from position_embedding import positionalencoding2d
import h5py
import math
from timm.layers import trunc_normal_
from typing import Optional, Tuple, Callable
from models.utils import RMSNorm

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


    
def read_features(feat_path):
    with h5py.File(feat_path,'r') as hdf5_file:
        features = hdf5_file['features'][:] # num_rois,84,1024
    return torch.from_numpy(features)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



def eager_attention_forward(
    module: nn.Module, # 注意：需要传入 Attention 模块实例
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs, 
):
    # K, V 的形状预期是 (B, num_kv_heads, N, D)
    # Query 的形状预期是 (B, num_heads, N, D)
    key_states = repeat_kv(key, module.num_key_value_groups)   # (B, H, Nk, D)
    value_states = repeat_kv(value, module.num_key_value_groups) # (B, H, Nk, D)

    # 计算 Attention Scores
    # query: (B, H, Nq, D), key_states.transpose: (B, H, D, Nk)
    # attn_weights: (B, H, Nq, Nk)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # Dropout
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # Weighted Value
    # (B, H, Nq, Nk) @ (B, H, Nk, D) -> (B, H, Nq, D)
    attn_output = torch.matmul(attn_weights, value_states)
    # Transpose 输出以匹配常见格式 (B, Nq, H, D) 
    attn_output = attn_output.transpose(1, 2).contiguous() 
    # 返回未 transpose 的结果和 attention weights
    return attn_output, attn_weights

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., **kwargs):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)


        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.norm = RMSNorm(dim_head)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(self.norm(q), self.norm(k).transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

'''
Modified Self-Attention module:
    add relative positional bias to self-attention: softmax(q*k+rel_pos)
    Refer to MaxViT
args:
    dim: input feature dimensions
    heads: number of attention heads
    dim_head: feature dimensions of each head
    dropout: dropout layer (default = 0, no dropout)
    have_cls_token: whether the module has class token, if false, use mean pooling for downstream task
    kwargs:
        window_size: Windos size for computing relative position bias. For instance,a ROI at 20x has 
            8*8=64 patches (each patch can be regarded as a pixel), window_size is 8.
        shared_pe: whether to share relative position embedding across all the heads
'''
class Rel_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,
                 have_cls_token=True, **kwargs):
        super().__init__()

        self.have_cls_token = have_cls_token
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        num_rel_position = 2*kwargs['window_size'] - 1 
       
        h_range = torch.arange(kwargs['window_size'])
        w_range = torch.arange(kwargs['window_size'])
        grid_x, grid_y = torch.meshgrid(h_range, w_range, indexing='ij')
        grid = torch.stack((grid_x, grid_y))
        grid = rearrange(grid, 'c h w -> c (h w)')
        grid = (grid[:,:,None]-grid[:,None,:]) + (kwargs['window_size']-1)
        bias_indices = (grid * torch.tensor([1,
                                                  num_rel_position])[:,None,None]).sum(dim=0)
        self.register_buffer("bias_indices", bias_indices)

        if kwargs['shared_pe'] == True:
            self.rel_pe = nn.Embedding(num_rel_position**2, 1)
        else: 
            self.rel_pe = nn.Embedding(num_rel_position**2, heads)

        trunc_normal_(self.rel_pe.weight, std=0.02)
        
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.norm = RMSNorm(dim_head)
        self.dropout = nn.Dropout(dropout)

        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        rel_position_bias = self.rel_pe(self.bias_indices)
        rel_position_bias = rearrange(rel_position_bias, 'i j h -> () h i j')

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(self.norm(q), self.norm(k).transpose(-1, -2)) * self.scale
        #dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.have_cls_token: 
            dots[:,:,1:,1:] += rel_position_bias
        else:
            dots += rel_position_bias

        attn = self.attend(dots)   
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


'''
Transformer block
args:
    dim: input feature dimension
    depth: the depth (number of Transformer layers) of the Transformer block
    heads: number of heads
    dim_head: feature dimension of each head
    mlp_dim: dimension of hidden layer feature in FeedForward module
    attn_type: the type of self-attention layer ('sa': normal self-attention, 'rel_sa':relative self-attention)
    shared_pe: whether to share relative position embedding across all the heads
    window_size: Windos size for computing relative position bias.
    have_cls_token: whether the module has class token, if false, use mean pooling for downstream task
'''
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,
                 attn_type='sa', shared_pe=None, window_size=None,
                 have_cls_token=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        if attn_type == 'sa':
            attn_layer = Attention
        elif attn_type == 'rel_sa':
            attn_layer = Rel_Attention
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn_layer(dim, heads = heads, dim_head =dim_head, 
                                        dropout = dropout,
                                        shared_pe=shared_pe,
                                        window_size=window_size,
                                        have_cls_token=have_cls_token)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


'''
ViT model
args:
    num_patches: the number of patches (the number of tokens)
    patch_dim: dimension of input tokens
    pool: whether to use cls token. 'cls': use cls token. 'mean': no cls token, use mean poolling.
    position = whether the positinoal encoding is learnable. 'learnable' or 'fixed'
    other arguments are the same with Transformer
'''
class ViT(nn.Module):
    def __init__(self, num_patches, patch_dim, dim, depth, heads, mlp_dim, 
                 pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.,
                 position='learnabel'):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_dim = patch_dim
        self.dim = dim
        self.to_patch_embedding = nn.Sequential(
            #nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            #nn.LayerNorm(dim)
        )
        if position=='learnable':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        elif position == 'fixed':
            pos_emb_20 = positionalencoding2d(dim, 8, 8)
            pos_emb_10 = positionalencoding2d(dim, 4, 4, 2)[1:]
            pos_emb_5 = positionalencoding2d(dim, 2, 2, 4)[1:]
            self.pos_embedding = torch.cat([pos_emb_20, pos_emb_10,
                                            pos_emb_5]).unsqueeze(0).cuda()
            print(self.pos_embedding.shape)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        if self.patch_dim != self.dim:
            x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


'''
Different versions of multi-scale self-attention network:
    different size of input ROI:
        PyramidViT_0: ROI size is 2048 (at 20x magnification level) 
        PyramidViT_1: ROI size is 1024 (at 20x magnification level) 
        PyramidViT_2: ROI size is 512 (at 20x magnification level) 
    
    PyramidViT_SingleScale: singel-scale model

    PyramidViT_wo_interscale: there is no inter-scale self-attention modules
'''

'''
PyramidViT_0:
    Input ROI has size of 2048 (at 20x). The number of tokens (patches) at 20x is 64. The size of each patch is 256x256.
    args:
        embed_weights: weight coefficient of instance embedding at each magnificant level (20x,10x,5x)
            List with the length of 3 or 'None' (learnable weights).
        depths: depth of transformer block at each scale.
            List with the length of 5:
                [intra-scale SA at 20x, 
                inter-scale between 20x~10x,
                intra-scale SA at 10x, 
                inter-scale between 10x~5x, 
                intra-scale SA at 5x]
        ape: whether to use positional encoding
'''
class PyramidViT_0(nn.Module):
    def __init__(self, num_patches =84, embed_weights =[0.3333,0.3333,0.3333] , patch_dim = 1024, dim=256, depths=[2,2,2,2,2], heads=4,
                 mlp_dim=512, pool='cls', dim_head = 64, dropout = 0., emb_dropout = 0.,
                 ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.patch_dim = patch_dim #dim of features extracted from ResNet
        self.dim = dim
        self.ape = ape
        self.embed_weights = embed_weights
        self.to_patch_embedding_20 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            #nn.LayerNorm(dim)
        )
        self.to_patch_embedding_10 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            #nn.LayerNorm(dim)
        )
        self.to_patch_embedding_5 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            #nn.LayerNorm(dim)
        )

        if ape:
            addition = 1 if pool == 'cls' else 0        #### pos embedding for cls_token
            self.pos_emb_20 = nn.Parameter(torch.zeros(1, 64+addition, dim))
            trunc_normal_(self.pos_emb_20, std=0.02)
            self.pos_emb_10 = nn.Parameter(torch.zeros(1, 16+addition, dim))
            trunc_normal_(self.pos_emb_10, std=0.02)
            self.pos_emb_5 = nn.Parameter(torch.zeros(1, 4+addition, dim))
            trunc_normal_(self.pos_emb_5, std=0.02)

        have_cls_token = False
        if pool == 'cls':
            have_cls_token = True
            self.cls_token_20 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_10 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_5 = nn.Parameter(torch.randn(1, 1, dim))
            trunc_normal_(self.cls_token_20, std=0.02)
            trunc_normal_(self.cls_token_10, std=0.02)
            trunc_normal_(self.cls_token_5, std=0.02)


        self.dropout = nn.Dropout(emb_dropout)

        assert len(depths) == 5
        self.transformer_20 = Transformer(dim, depths[0], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 8,
                                          have_cls_token)
        self.transformer_20_to_10 = Transformer(dim, depths[1], heads, dim_head,
                                                mlp_dim, dropout, 'sa')     # no need to set have_cls_token
        self.transformer_10 = Transformer(dim, depths[2], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 4,
                                          have_cls_token)
        self.transformer_10_to_5 = Transformer(dim, depths[3], heads, dim_head,
                                               mlp_dim, dropout, 'sa')
        self.transformer_5 = Transformer(dim, depths[4], heads, dim_head, mlp_dim, dropout,
                                         attn_type, shared_pe, 2,
                                         have_cls_token)

        if embed_weights == None:
            print('learnable embedding weights')
            self.learned_weights = nn.Parameter(torch.Tensor(3,1))
            ## init
            nn.init.kaiming_uniform(self.learned_weights,a=math.sqrt(5))

        self.ms_dropout = nn.Dropout(dropout)



    def forward(self, x):
        b, _, _ = x.shape
        
        
        x_20 = x[:, :64, :]     # b 64 c
        x_10 = x[:, 64:80, :]   # b 16 c
        x_5 = x[:, 80:84, :]    # b 4 c
        
        if self.patch_dim != self.dim:
            x_20 = self.to_patch_embedding_20(x_20)
            x_10 = self.to_patch_embedding_10(x_10)
            x_5 = self.to_patch_embedding_5(x_5)

        if self.pool == 'cls':
            cls_token_20 = repeat(self.cls_token_20, '() n d -> b n d', b=b)  #b 1 c
            x_20 = torch.cat((cls_token_20, x_20), dim=1)   #b 65 c
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)   
            x_20 = self.transformer_20(x_20)    # b 65 c
            x_20_cls_token = x_20[:,0,:]      # b c
            x_20 = x_20[:,1:,:]     #b 64 c

            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=4, h2=2, w1=4, w2=2)    # 16b 4 c
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)  # 16b 1 c
            x_20_10 = torch.cat((x_10, x_20), dim=1)    # 16b 5 c
            x_20_10 = self.transformer_20_to_10(x_20_10) # 16b 5 c
            x_10 = x_20_10[:, 0:1, :]       # 16b 1 c

            cls_token_10 = repeat(self.cls_token_10, '() n d -> b n d', b=b)  #b 1 c
            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)   #b 16 c
            x_10 = torch.cat((cls_token_10, x_10), dim=1)   #b 17 c
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)   
            x_10 = self.transformer_10(x_10)    # b 17 c
            x_10_cls_token = x_10[:,0,:]    # b c
            x_10 = x_10[:,1:,:]     # b 16 c

            x_10 = rearrange(x_10, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=2, h2=2, w1=2, w2=2)    # 4b 4 c
            x_5 = rearrange(x_5, 'b (n m) c -> (b n) m c', m=1)  # 4b 1 c
            x_10_5 = torch.cat((x_5, x_10), dim=1)    # 4b 5 c
            x_10_5 = self.transformer_10_to_5(x_10_5) # 4b 5 c
            x_5 = x_10_5[:, 0:1, :]       # 4b 1 c

            cls_token_5 = repeat(self.cls_token_5, '() n d -> b n d', b=b)  #b 1 c
            x_5 = rearrange(x_5, '(b n) m c -> b (n m) c', b=b)   #b 4 c
            x_5 = torch.cat((cls_token_5, x_5), dim=1)   #b 5 c
            if self.ape:
                x_5 += self.pos_emb_5
            x_5 = self.dropout(x_5)   
            x_5 = self.transformer_5(x_5)    # b 5 c
            x_5_cls_token = x_5[:,0,:]      # b c

        elif self.pool == 'mean':
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)   
            x_20 = self.transformer_20(x_20)    # b 64 c
            x_20_cls_token = x_20.mean(dim=1)      # b c

            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=4, h2=2, w1=4, w2=2)    # 16b 4 c
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)  # 16b 1 c
            x_20_10 = torch.cat((x_10, x_20), dim=1)    # 16b 5 c
            x_20_10 = self.transformer_20_to_10(x_20_10) # 16b 5 c
            x_10 = x_20_10[:, 0:1, :]       # 16b 1 c

            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)   #b 16 c
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)   
            x_10 = self.transformer_10(x_10)    # b 16 c
            x_10_cls_token = x_10.mean(dim=1)      # b c

            x_10 = rearrange(x_10, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=2, h2=2, w1=2, w2=2)    # 4b 4 c
            x_5 = rearrange(x_5, 'b (n m) c -> (b n) m c', m=1)  # 4b 1 c
            x_10_5 = torch.cat((x_5, x_10), dim=1)    # 4b 5 c
            x_10_5 = self.transformer_10_to_5(x_10_5) # 4b 5 c
            x_5 = x_10_5[:, 0:1, :]       # 4b 1 c

            x_5 = rearrange(x_5, '(b n) m c -> b (n m) c', b=b)   #b 4 c
            if self.ape:
                x_5 += self.pos_emb_5
            x_5 = self.dropout(x_5)   
            x_5 = self.transformer_5(x_5)    # b 4 c
            x_5_cls_token = x_5.mean(dim=1)      # b c
        
        if self.embed_weights == None:
            learned_weights = torch.softmax(self.learned_weights,dim=0)

            x = learned_weights[0]*x_5_cls_token + learned_weights[1]*x_10_cls_token + learned_weights[2]*x_20_cls_token


        else:
            x_stack = torch.stack((self.embed_weights[0]*x_5_cls_token, 
                                   self.embed_weights[1]*x_10_cls_token, 
                                   self.embed_weights[2]*x_20_cls_token))
            x = torch.sum(x_stack,dim=0) # b c
        
        
        return x


'''
PyramidViT_1:
    Input ROI has size of 1024 (at 20x). The number of tokens (patches) at 20x is 16. The size of each patch is 256x256.
    args:
        embed_weights: weight coefficient of instance embedding at each magnificant level (20x,10x,5x)
            List with the length of 3 or 'None' (learnable weights).
        depths: depth of transformer block at each scale.
            List with the length of 5:
                [intra-scale SA at 20x, 
                inter-scale between 20x~10x,
                intra-scale SA at 10x, 
                inter-scale between 10x~5x, 
                intra-scale SA at 5x]
        ape: whether to use positional encoding
'''
class PyramidViT_1(nn.Module):
    def __init__(self, num_patches, embed_weights, patch_dim, dim, depths, heads=4,
                 mlp_dim=512, pool='cls', dim_head = 64, dropout = 0., emb_dropout = 0.,
                 ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.patch_dim = patch_dim #dim of features extracted from ResNet
        self.dim = dim
        self.ape = ape
        self.embed_weights = embed_weights
        self.to_patch_embedding_20 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_10 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_5 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

        if ape:
            addition = 1 if pool == 'cls' else 0        #### pos embedding for cls_token
            self.pos_emb_20 = nn.Parameter(torch.zeros(1, 16+addition, dim))
            trunc_normal_(self.pos_emb_20, std=0.02)
            self.pos_emb_10 = nn.Parameter(torch.zeros(1, 4+addition, dim))
            trunc_normal_(self.pos_emb_10, std=0.02)
            self.pos_emb_5 = nn.Parameter(torch.zeros(1, 1 + addition, dim))
            trunc_normal_(self.pos_emb_5, std = 0.02)

        have_cls_token = False
        if pool == 'cls':
            have_cls_token = True
            self.cls_token_20 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_10 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_5 = nn.Parameter(torch.randn(1, 1, dim))
            trunc_normal_(self.cls_token_20, std=0.02)
            trunc_normal_(self.cls_token_10, std=0.02)
            trunc_normal_(self.cls_token_5, std=0.02)


        self.dropout = nn.Dropout(emb_dropout)

        assert len(depths) == 5
        self.transformer_20 = Transformer(dim, depths[0], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 4,
                                          have_cls_token)
        self.transformer_20_to_10 = Transformer(dim, depths[1], heads, dim_head,
                                               mlp_dim, dropout, 'sa')
        self.transformer_10 = Transformer(dim, depths[2], heads, dim_head, mlp_dim, dropout,
                                         attn_type, shared_pe, 2,
                                         have_cls_token)
        self.transformer_10_to_5 = Transformer(dim, depths[3], heads, dim_head,
                                               mlp_dim, dropout, 'sa')
        self.transformer_5 = Transformer(dim, depths[4], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 1,
                                          have_cls_token)
        

        if embed_weights == None:
            print('learnable embedding weights')
            self.learned_weights = nn.Parameter(torch.Tensor(3,1))
            nn.init.kaiming_uniform(self.learned_weights,a=math.sqrt(5))
        self.ms_dropout = nn.Dropout(dropout)



    def forward(self, x):
        b, _, _ = x.shape
        
        
        x_20 = x[:, :16, :]   # b 16 c
        x_10 = x[:, 16:20, :]  # b 4 c
        x_5 = x[:,20:,:]    # b 1 c 
        
        if self.patch_dim != self.dim:
            x_20 = self.to_patch_embedding_20(x_20)
            x_10 = self.to_patch_embedding_10(x_10)
            x_5 = self.to_patch_embedding_5(x_5)

        if self.pool == 'cls':

            cls_token_20 = repeat(self.cls_token_20, '() n d -> b n d', b=b)  #b 1 c
            x_20 = torch.cat((cls_token_20, x_20), dim=1)   #b 17 c
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)   
            x_20 = self.transformer_20(x_20)    # b 17 c
            x_20_cls_token = x_20[:,0,:]    # b c
            x_20 = x_20[:,1:,:]     # b 16 c

            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=2, h2=2, w1=2, w2=2)    # 4b 4 c
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)  # 4b 1 c
            x_20_10 = torch.cat((x_10, x_20), dim=1)    # 4b 5 c
            x_20_10 = self.transformer_20_to_10(x_20_10) # 4b 5 c
            x_10 = x_20_10[:, 0:1, :]       # 4b 1 c

            cls_token_10 = repeat(self.cls_token_10, '() n d -> b n d', b=b)  #b 1 c
            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)   #b 4 c
            x_10 = torch.cat((cls_token_10, x_10), dim=1)   #b 5 c
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)   
            x_10 = self.transformer_10(x_10)    # b 5 c
            x_10_cls_token = x_10[:,0,:]      # b c
            x_10 = x_10[:,1:,:]       # b 4 c

            x_10 = rearrange(x_10, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                         h1=1, h2=2, w1=1, w2=2)    # b 4 c
            x_5 = rearrange(x_5, 'b (n m) c -> (b n) m c', m=1)  # b 1 c
            x_10_5 = torch.cat((x_5, x_10), dim=1)    # b 5 c
            x_10_5 = self.transformer_10_to_5(x_10_5) # b 5 c
            x_5 = x_10_5[:, 0:1, :]       # b 1 c

            cls_token_5 = repeat(self.cls_token_5, '() n d -> b n d', b=b)
            x_5 = rearrange(x_5, '(b n) m c -> b (n m) c', b=b)   #b 1 c
            x_5 = torch.cat((cls_token_5, x_5), dim=1)   # b 2 c
            if self.ape:
                x_5 += self.pos_emb_5
            x_5 = self.dropout(x_5)   
            x_5 = self.transformer_5(x_5)    # b 2 c
            x_5_cls_token = x_5[:,0,:]      # b c

        elif self.pool == 'mean':
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)   
            x_20 = self.transformer_20(x_20)    # b 16 c
            x_20_cls_token = x_20.mean(dim=1)      # b c

            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=2, h2=2, w1=2, w2=2)    # 4b 4 c
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)  # 4b 1 c
            x_20_10 = torch.cat((x_10, x_20), dim=1)    # 4b 5 c
            x_20_10 = self.transformer_10_to_10(x_20_10) # 4b 5 c
            x_10 = x_20_10[:, 0:1, :]       # 4b 1 c

            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)   #b 4 c
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)   
            x_10 = self.transformer_10(x_10)    # b 4 c
            x_10_cls_token = x_10.mean(dim=1)      # b c
        
            x_10 = rearrange(x_10, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                         h1=1, h2=2, w1=1, w2=2)    # b 4 c
            x_5 = rearrange(x_5, 'b (n m) c -> (b n) m c', m=1)  # b 1 c
            x_10_5 = torch.cat((x_5, x_10), dim=1)    # b 5 c
            x_10_5 = self.transformer_10_to_5(x_10_5) # b 5 c
            x_5 = x_10_5[:, 0:1, :]       # b 1 c

            x_5 = rearrange(x_5, '(b n) m c -> b (n m) c', b=b)   #b 1 c
            if self.ape:
                x_5 += self.pos_emb_5
            x_5 = self.dropout(x_5)   
            x_5 = self.transformer_5(x_5)   # b 1 c
            x_5_cls_token = x_5.mean(dim=1)      # b c

        if self.embed_weights == None:

            learned_weights = torch.softmax(self.learned_weights,dim=0)

            x = learned_weights[0]*x_5_cls_token + learned_weights[1]*x_10_cls_token + learned_weights[2]*x_20_cls_token


       
        else:
            x_stack = torch.stack((self.embed_weights[0]*x_5_cls_token, 
                                   self.embed_weights[1]*x_10_cls_token, 
                                   self.embed_weights[2]*x_20_cls_token))
            x = torch.sum(x_stack,dim=0) # b c

        
        return x


'''
PyramidViT_2:
    Input ROI has size of 512 (at 20x). The number of tokens (patches) at 20x is 4. The size of each patch is 256x256.
    args:
        embed_weights: weight coefficient of instance embedding at each magnificant level (20x,10x), no 5x because number of patches at 10x is already 1.
            List with the length of 2 or 'None' (learnable weights).
        depths: depth of transformer block at each scale.
            List with the length of 3:
                [intra-scale SA at 20x, 
                inter-scale between 20x~10x,
                intra-scale SA at 10x]
        ape: whether to use positional encoding
'''
class PyramidViT_2(nn.Module):
    def __init__(self, num_patches, embed_weights, patch_dim, dim, depths, heads=4,
                 mlp_dim=512, pool='cls', dim_head = 64, dropout = 0., emb_dropout = 0.,
                 ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.patch_dim = patch_dim #dim of features extracted from ResNet
        self.dim = dim
        self.ape = ape
        self.embed_weights = embed_weights
        self.to_patch_embedding_20 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_10 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )


        if ape:
            addition = 1 if pool == 'cls' else 0        #### pos embedding for cls_token
            self.pos_emb_20 = nn.Parameter(torch.zeros(1, 4+addition, dim))
            trunc_normal_(self.pos_emb_20, std=0.02)
            self.pos_emb_10 = nn.Parameter(torch.zeros(1, 1+addition, dim))
            trunc_normal_(self.pos_emb_10, std=0.02)


        have_cls_token = False
        if pool == 'cls':
            have_cls_token = True
            self.cls_token_20 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_10 = nn.Parameter(torch.randn(1, 1, dim))            
            trunc_normal_(self.cls_token_20, std=0.02)
            trunc_normal_(self.cls_token_10, std=0.02)



        self.dropout = nn.Dropout(emb_dropout)

        assert len(depths) == 3
        self.transformer_20 = Transformer(dim, depths[0], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 2,
                                          have_cls_token)
        self.transformer_20_to_10 = Transformer(dim, depths[1], heads, dim_head,
                                               mlp_dim, dropout, 'sa')
        self.transformer_10 = Transformer(dim, depths[2], heads, dim_head, mlp_dim, dropout,
                                         attn_type, shared_pe, 1,
                                         have_cls_token)
        

        if embed_weights == None:
            print('learnable embedding weights')
            self.learned_weights = nn.Parameter(torch.Tensor(2,1))
            nn.init.kaiming_uniform(self.learned_weights,a=math.sqrt(5))
        self.ms_dropout = nn.Dropout(dropout)



    def forward(self, x):
        b, _, _ = x.shape
        
        
        x_20 = x[:, :4, :]   # b 4 c
        x_10 = x[:, 4:, :]  # b 1 c

        
        if self.patch_dim != self.dim:
            x_20 = self.to_patch_embedding_20(x_20)
            x_10 = self.to_patch_embedding_10(x_10)

        if self.pool == 'cls':

            cls_token_20 = repeat(self.cls_token_20, '() n d -> b n d', b=b)  #b 1 c
            x_20 = torch.cat((cls_token_20, x_20), dim=1)   #b 4 c
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)   
            x_20 = self.transformer_20(x_20)    # b 5 c
            x_20_cls_token = x_20[:,0,:]    # b c
            x_20 = x_20[:,1:,:]     # b 4 c

            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=1, h2=2, w1=1, w2=2)    # b 4 c
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)  # b 1 c
            x_20_10 = torch.cat((x_10, x_20), dim=1)    # b 5 c
            x_20_10 = self.transformer_20_to_10(x_20_10) # b 5 c
            x_10 = x_20_10[:, 0:1, :]       # b 1 c

            cls_token_10 = repeat(self.cls_token_10, '() n d -> b n d', b=b)  #b 1 c
            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)   #b 1 c
            x_10 = torch.cat((cls_token_10, x_10), dim=1)   #b 2 c
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)   
            x_10 = self.transformer_10(x_10)    # b 2 c
            x_10_cls_token = x_10[:,0,:]      # b c

        elif self.pool == 'mean':
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)   
            x_20 = self.transformer_20(x_20)    # b 4 c
            x_20_cls_token = x_20.mean(dim=1)      # b c

            x_20 = rearrange(x_20, 'b (h1 h2 w1 w2) c -> (b h1 w1) (h2 w2) c',
                             h1=2, h2=2, w1=2, w2=2)    # b 4 c
            x_10 = rearrange(x_10, 'b (n m) c -> (b n) m c', m=1)  # b 1 c
            x_20_10 = torch.cat((x_10, x_20), dim=1)    # b 5 c
            x_20_10 = self.transformer_10_to_10(x_20_10) # b 5 c
            x_10 = x_20_10[:, 0:1, :]       # b 1 c

            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)   #b 4 c
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)   
            x_10 = self.transformer_10(x_10)    # b 1 c
            x_10_cls_token = x_10.mean(dim=1)      # b c
        

        if self.embed_weights == None:
            learned_weights = torch.softmax(self.learned_weights,dim=0)

            x = learned_weights[0]*x_10_cls_token + learned_weights[1]*x_20_cls_token
            #print('x', x.shape)

       
        else:
            x_stack = torch.stack((self.embed_weights[0]*x_10_cls_token, 
                                   self.embed_weights[1]*x_20_cls_token))
            x = torch.sum(x_stack,dim=0) # b c

        
        return x


'''
PyramidViT_SingleScale:
    Single-scale model
    args:
        embed_weights: useless
        depths: depth of transformer block at each scale.
        ape: whether to use positional encoding
'''
class PyramidViT_SingleScale(nn.Module):
    def __init__(self, num_patches, embed_weights, patch_dim, dim, depths, level=0, heads=4,
                 mlp_dim=512, pool='cls', dim_head = 64, dropout = 0., emb_dropout = 0.,
                 ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.patch_dim = patch_dim #dim of features extracted from ResNet
        self.dim = dim
        self.ape = ape
        self.level = level
        self.embed_weights = embed_weights

        self.to_patch_embedding = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.patch_idx = [0,64,80,84] 


        patch_levels = [64,16,4]
        patch_length = [8,4,2]
        

        if ape:
            addition = 1 if pool == 'cls' else 0        #### pos embedding for cls_token
            self.pos_emb = nn.Parameter(torch.zeros(1, patch_levels[self.level]+addition, dim))
            trunc_normal_(self.pos_emb, std=0.02)

        have_cls_token = False
        if pool == 'cls':
            have_cls_token = True
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            trunc_normal_(self.cls_token, std=0.02)


        self.dropout = nn.Dropout(emb_dropout)

        assert len(depths) == 5
        self.transformer = Transformer(dim, depths[0], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, patch_length[self.level],
                                          have_cls_token)

        self.ms_dropout = nn.Dropout(dropout)



    def forward(self, x):
        b, _, _ = x.shape
        
        
        x_level = x[:, self.patch_idx[self.level]:self.patch_idx[self.level+1], :]     # b n c
        
        if self.patch_dim != self.dim:
            x_level = self.to_patch_embedding(x_level)

        if self.pool == 'cls':
            cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
            x_level = torch.cat((cls_token, x_level), dim=1)
            if self.ape:
                x_level += self.pos_emb
            x_level = self.dropout(x_level)   
            x_level = self.transformer(x_level)    # b 65 c
            x_level_cls_token = x_level[:,0,:]      # b c

        elif self.pool == 'mean':
            if self.ape:
                x_level += self.pos_emb
            x_level = self.dropout(x_level)   
            x_level = self.transformer(x_level)    # b 64 c
            x_level_cls_token = x_level.mean(dim=1)      # b c

        

        x_level = torch.cat((x_level_cls_token,), dim=1)
        
        return x_level


'''
PyramidViT_wo_interscale:
    There are no inter-scale self-attention modules.
    args:
        embed_weights: weight coefficient of instance embedding at each magnificant level (20x,10x,5x)
            List with the length of 3 or 'None' (learnable weights).
        depths: depth of transformer block at each scale.
            List with the length of 3:
                [intra-scale SA at 20x, 
                intra-scale SA at 10x, 
                intra-scale SA at 5x]
        ape: whether to use positional encoding
'''
class PyramidViT_wo_interscale(nn.Module):
    def __init__(self, num_patches, embed_weights, patch_dim, dim, depths, heads=4,
                 mlp_dim=512, pool='cls', dim_head = 64, dropout = 0., emb_dropout = 0.,
                 ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.ape = ape
        
        self.patch_dim = patch_dim #dim of features extracted from ResNet
        self.dim = dim
        self.embed_weights = embed_weights
        self.to_patch_embedding_20 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_10 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_5 = nn.Sequential(
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

        if ape:
            addition = 1 if pool == 'cls' else 0        #### pos embedding for cls_token
            self.pos_emb_20 = nn.Parameter(torch.zeros(1, 64+addition, dim))
            trunc_normal_(self.pos_emb_20, std=0.02)
            self.pos_emb_10 = nn.Parameter(torch.zeros(1, 16+addition, dim))
            trunc_normal_(self.pos_emb_10, std=0.02)
            self.pos_emb_5 = nn.Parameter(torch.zeros(1, 4+addition, dim))
            trunc_normal_(self.pos_emb_5, std=0.02)
        
        have_cls_token = False
        if pool == 'cls':
            have_cls_token = True
            self.cls_token_20 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_10 = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_5 = nn.Parameter(torch.randn(1, 1, dim))
            trunc_normal_(self.cls_token_20, std=0.02)
            trunc_normal_(self.cls_token_10, std=0.02)
            trunc_normal_(self.cls_token_5, std=0.02)
        
        self.dropout = nn.Dropout(emb_dropout)

        assert len(depths) == 3
        self.transformer_20 = Transformer(dim, depths[0], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 8,
                                          have_cls_token)
        # self.transformer_20_to_10 = Transformer(dim, depths[1], heads, dim_head,
        #                                         mlp_dim, dropout, 'sa')     # no need to set have_cls_token
        self.transformer_10 = Transformer(dim, depths[1], heads, dim_head, mlp_dim, dropout,
                                          attn_type, shared_pe, 4,
                                          have_cls_token)
        # self.transformer_10_to_5 = Transformer(dim, depths[3], heads, dim_head,
        #                                        mlp_dim, dropout, 'sa')
        self.transformer_5 = Transformer(dim, depths[2], heads, dim_head, mlp_dim, dropout,
                                         attn_type, shared_pe, 2,
                                         have_cls_token)


        if embed_weights == None:
            print('learnable embedding weights')
            #'''
            self.ms_attn = nn.Sequential(
                mlpmixer(dim=dim, num_patches=3,
                         dim_expansion_factor=2, dropout=dropout),
                nn.Linear(dim, dim//2),
                nn.Tanh(),
                nn.Linear(dim//2, 3)
            )

        self.ms_dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, _, _ = x.shape
        

        
        x_20 = x[:, :64, :]     # b 64 c
        x_10 = x[:, 64:80, :]   # b 16 c
        x_5 = x[:, 80:84, :]    # b 4 c
        
        if self.patch_dim != self.dim:
            x_20 = self.to_patch_embedding_20(x_20)
            x_10 = self.to_patch_embedding_10(x_10)
            x_5 = self.to_patch_embedding_5(x_5)

        if self.pool == 'cls':
            cls_token_20 = repeat(self.cls_token_20, '() n d -> b n d', b=b) #b 1 c
            x_20 = torch.cat((cls_token_20, x_20), dim=1)   # b 65 c
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)   
            x_20 = self.transformer_20(x_20)    # b 65 c
            x_20_cls_token = x_20[:,0,:]      # b c
            x_20 = x_20[:, 1:, :]       # b 64 c
            

            cls_token_10 = repeat(self.cls_token_10, '() n d -> b n d', b=b)
            x_10 = rearrange(x_10, '(b n) m c -> b (n m) c', b=b)   #b 16 c
            x_10 = torch.cat((cls_token_10, x_10), dim=1)   # b 17 c
            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)   
            x_10 = self.transformer_10(x_10)    # b 17 c
            x_10_cls_token = x_10[:,0,:]      # b c
            x_10 = x_10[:, 1:, :]       # b 16 c
            

            cls_token_5 = repeat(self.cls_token_5, '() n d -> b n d', b=b)
            x_5 = rearrange(x_5, '(b n) m c -> b (n m) c', b=b)   #b 4 c
            x_5 = torch.cat((cls_token_5, x_5), dim=1)   # b 5 c
            if self.ape:
                x_5 += self.pos_emb_5
            x_5 = self.dropout(x_5)   
            x_5 = self.transformer_5(x_5)    # b 5 c
            x_5_cls_token = x_5[:,0,:]      # b c
        
        elif self.pool == 'mean':
            if self.ape:
                x_20 += self.pos_emb_20
            x_20 = self.dropout(x_20)   
            x_20 = self.transformer_20(x_20)    # b 65 c
            x_20_cls_token = x_20.mean(dim=1)      # b c
            

            if self.ape:
                x_10 += self.pos_emb_10
            x_10 = self.dropout(x_10)   
            x_10 = self.transformer_10(x_10)    # b 17 c
            x_10_cls_token = x_10.mean(dim=1)      # b c
            

            if self.ape:
                x_5 += self.pos_emb_5
            x_5 = self.dropout(x_5)   
            x_5 = self.transformer_5(x_5)    # b 5 c
            x_5_cls_token = x_5.mean(dim=1)      # b c
       

        if self.embed_weights == None:
            x_stack = torch.stack([x_20_cls_token, x_10_cls_token,
                                   x_5_cls_token], dim=1)       ## b 3 c
            w = self.ms_attn(x_stack)       ## b 3
            w = torch.softmax(w, dim=1)
            x = torch.einsum('b m, b m c -> b c', w, x_stack)

        else:
            x_stack = torch.stack((self.embed_weights[0]*x_5_cls_token, 
                                   self.embed_weights[1]*x_10_cls_token, 
                                   self.embed_weights[2]*x_20_cls_token))
            x = torch.sum(x_stack,dim=0) # b c

        return x




class deepmil(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.):
        super(deepmil, self).__init__()

        self.dim_trans = nn.Sequential(
            #nn.Linear(512*64, dim),
            nn.Linear(input_dim, hidden_dim*2),
            nn.GELU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(inplace=True)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.Tanh(),
            nn.Linear(input_dim//2, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dim_trans(x)       #b n 512
        a = self.attention(x)       #b n 1
        a = F.softmax(a, dim=1)
        
        x = einsum('b n d, b n a -> b a d', x, a).squeeze(1) #b,512
        x = self.dropout(x)

        return x


class mlpmixer(nn.Module):
    def __init__(self, dim=512, num_patches=20, dim_expansion_factor=2,
                 dropout=0.):
        super(mlpmixer, self).__init__()
        
        self.token_mixer = nn.Sequential(
            RMSNorm(dim),
            nn.Conv1d(num_patches, num_patches*dim_expansion_factor, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            #nn.Dropout(0.25),
            nn.Conv1d(num_patches*dim_expansion_factor, num_patches, 1),
            #nn.Dropout(dropout)
        )

        self.channel_mixer = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim*dim_expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            #nn.Dropout(0.25),
            nn.Linear(dim*dim_expansion_factor, dim),
            #nn.Dropout(dropout)
        )

    def forward(self, x):
        idn1 = x
        x = self.token_mixer(x)
        x = x + idn1
        idn2 = x
        x = self.channel_mixer(x)
        x = x + idn2

        return torch.mean(x, dim=1)

'''
Class-specific gated attention net of CLAM. For instance aggregation.
args:
    input_dim: input feature dimension
    out_dim: out feature dimension
    n_classes: number of classes
'''
# modified by CLAM
class Attention_net_gated(nn.Module):
    def __init__(self,input_dim = 256,out_dim = 256,n_classes = 1,dropout=0):
        super(Attention_net_gated,self).__init__()
        self.attention_a = [RMSNorm(input_dim),nn.Linear(input_dim,out_dim),nn.Tanh()]
        self.attention_b = [RMSNorm(input_dim),nn.Linear(input_dim,out_dim),nn.Sigmoid()]

        if dropout>0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
            #self.attention_a.append(nn.Dropout(0.2))
            #self.attention_b.append(nn.Dropout(0.2))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(out_dim, n_classes)
    
    def forward(self,x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A,x


'''
Class-specific normal attention net of CLAM. For instance aggregation.
args:
    input_dim: input feature dimension
    out_dim: out feature dimension
    n_classes: number of classes
'''
class Attention_net(nn.Module):
    def __init__(self,input_dim = 256,out_dim = 256,n_classes = 1,dropout=0):
        super(Attention_net,self).__init__()
        self.attention_a = [RMSNorm(input_dim), nn.Linear(input_dim,out_dim), nn.Tanh()]

        if dropout>0:
            self.attention_a.append(nn.Dropout(dropout))
            #self.attention_a.append(nn.Dropout(0.2))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        
        self.attention_c = nn.Linear(out_dim, n_classes)
    
    def forward(self,x):
        a = self.attention_a(x)
        A = self.attention_c(a)
        return A,x


class Classifier(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        return x

'''
ROAM: ROAM model main architecture
args:
    choose_num: the number of instances (topk) used in instance-level supervision
    num_patches: the number of patches (tokens)
    patch_dim: input patch dimension
    num_classes: the number of classes
    roi_level: size of ROI at 20x, (0:2048,1:1024,2:512)
    scale_type: 'ms' (multi-scale) or 'ss' (single scale)
    embed_weights: weight coefficient of instance embedding at each magnificant level (5x,10x,20x)
    dim: input feature dimension
    depths: depths (number of Transformer layers) of the Transformer block
    heads: number of heads
    mlp_dim: dimension of hidden layer feature
    not_interscale: False: there exists inter-scale self-attention module, True: there is no inter-scale self-attention module
    single_level: magnification scale of input ROI for single-scale model. (0:20x,1:10x,2:5x)
    dim_head: feature dimension of each head
    dropout: dropout of self-attentio module and feedforward module of transformer
    emb_dropout: dropout of multi-scale self-attention network
    attn_dropout: dropout of attention network in instance aggregation module
    pool: whether to use cls token. 'cls': use cls token. 'mean': no cls token
    ape: whether to use positional encoding
    attn_type: the type of self-attention layer ('sa': normal self-attention, 'rel_sa': relative self-attention)
    shared_pe: whether to share relative position embedding across all the heads
'''
class ROAM(nn.Module):
    def __init__(self, num_patches, patch_dim, 
                 num_classes, roi_level, scale_type,
                 embed_weights, dim, depths, heads, mlp_dim, 
                 not_interscale=False, single_level=0,
                 dim_head=64, dropout=0., emb_dropout=0., attn_dropout=0., 
                 pool='cls', ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()
        self.num_classes = num_classes
        self.roi_level = roi_level
        self.scale_type = scale_type
        self.embed_weights = embed_weights

        if self.scale_type == 'ms':
            if not_interscale:
                self.vit = PyramidViT_wo_interscale(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
            else:
                if self.roi_level == 0:
                    self.vit = PyramidViT_0(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
                elif self.roi_level == 1:
                    self.vit = PyramidViT_1(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
                else:
                    self.vit = PyramidViT_2(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
        else:
            self.vit = PyramidViT_SingleScale(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, level=single_level, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
        
        ## attention net
        self.att_net = Attention_net_gated(input_dim=dim,out_dim=dim//2,n_classes=num_classes,dropout=attn_dropout)
        
        self.roi_clf = Classifier(dim, num_classes)
        self.slide_clfs = nn.ModuleList([Classifier(dim, 1) for i in range(num_classes)])

        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        '''
        self.loss_fn = nn.CrossEntropyLoss()

    def get_pathomics_features(self,hidden_states:torch.Tensor):
        b,k, _, _ = hidden_states.shape
        hidden_states = rearrange(hidden_states, 'b k n d -> (b k) n d')
        features = self.vit(hidden_states)  #(batch_size * num_ROIs, dim)
        return features

    def forward(self,x,label=None,inst_level=False,grade=False):
        device = x.device
        b,k, _, _ = x.shape
        x = rearrange(x, 'b k n d -> (b k) n d')
        x = self.vit(x)         # (b*k dim) 
        x = rearrange(x, '(b k) d -> b k d', b=b)
        slide_logits = torch.empty(b, self.num_classes).float().to(device)
        inst_loss_all = 0.0
        for b_i in range(b):
            h = x[b_i] #n_rois,dim

            att_embed,_ = self.att_net(h) #k,n_classes
            att_embed = torch.transpose(att_embed,1,0) #n_classes,k
            att_embed = F.softmax(att_embed,dim=1) #n_classes,k

            total_inst_loss = 0.0
            tmp_topk = min(self.topk, k)
            ## instance-level supervision
            if inst_level:
                l = label[b_i]
                att_cur_cat = att_embed[l]
                in_top_p_idx = torch.topk(att_cur_cat, tmp_topk, dim=0)[1]
                in_top_p = h[in_top_p_idx]
                in_p_targets = torch.full((tmp_topk, ), l, device=device).long()

                logits = self.roi_clf(in_top_p)
                inst_loss = self.loss_fn(logits, in_p_targets)
                total_inst_loss += inst_loss
            

            h_weighted = torch.mm(att_embed,h) #n_classes,dim
            for c in range(self.num_classes):
                slide_logits[b_i,c] = self.slide_clfs[c](h_weighted[c])
            
            '''
            tmp_out = self.slide_clf(h_weighted)[0]
            #print(tmp_out.shape, slide_logits[b_i].shape)
            slide_logits[b_i] = tmp_out 
            '''
        return slide_logits,total_inst_loss




class ROAM_VIS(nn.Module):
    def __init__(self, *, choose_num, num_patches, patch_dim, 
                 num_classes, roi_level, scale_type,
                 embed_weights, dim, depths, heads, mlp_dim, 
                 not_interscale=False, single_level=0,
                 dim_head=64, dropout=0., emb_dropout=0., attn_dropout=0., 
                 pool='cls', ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()
        #self.aggr_method = aggr_method
        self.topk = choose_num
        self.num_classes = num_classes
        self.roi_level = roi_level
        self.scale_type = scale_type
        self.embed_weights = embed_weights

        if self.scale_type == 'ms':
            if not_interscale:
                self.vit = PyramidViT_wo_interscale(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
            else:
                if self.roi_level == 0:
                    self.vit = PyramidViT_0(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
                elif self.roi_level == 1:
                    self.vit = PyramidViT_1(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
                else:
                    self.vit = PyramidViT_2(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
        else:
            self.vit = PyramidViT_SingleScale(num_patches=num_patches, embed_weights = self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, level=single_level, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
        
        ## attention net
        self.att_net = Attention_net_gated(input_dim=dim,out_dim=dim//2,n_classes=num_classes,dropout=attn_dropout)
        
        self.roi_clf = Classifier(dim, num_classes)
        self.slide_clfs = nn.ModuleList([Classifier(dim, 1) for i in range(num_classes)])

        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        '''

        #self.loss_fn = SmoothTop1SVM(n_classes=2).cuda()
        self.loss_fn = nn.CrossEntropyLoss()


                
    '''
    args:
        vis: whether to use visualization mode
            False: normal mode, with output similar to that during ROAM training
                input: patch features extracted from pre-trained model
                output: slide-level prediciton and instance-level loss
        vis_mode: configure of visualization
            1: no instance aggregation is performed
                input: patch features extracted from pre-trained model
                output: instance-level representations
            2: 
                input: instance-level representations
                output: slide-level prediciton and attention scores of each ROI (instance)
            3:
                input: patch features extracted from pre-trained model
                output: slide-level prediciton and attention scores of each ROI (instance)
    '''
    def forward(self,x,label=None,inst_level=False,grade=False,vis=False,vis_mode=1):
        if not vis:
            device = x.device
            b,k, _, _ = x.shape
            x = rearrange(x, 'b k n d -> (b k) n d')
            x = self.vit(x)         # (b*k dim) 
            x = rearrange(x, '(b k) d -> b k d', b=b)
            slide_logits = torch.empty(b, self.num_classes).float().to(device)
            inst_loss_all = 0.0
            for b_i in range(b):
                h = x[b_i] #n_rois,dim
                #device = h.device

                att_embed,_ = self.att_net(h) #k,n_classes
                roi_attns = att_embed
                att_embed = torch.transpose(att_embed,1,0) #n_classes,k
                att_embed = F.softmax(att_embed,dim=1) #n_classes,k

                total_inst_loss = 0.0
                #inst_loss_list = []
                tmp_topk = min(self.topk, k)
                if inst_level:
                    l = label[b_i]
                    att_cur_cat = att_embed[l]
                    in_top_p_idx = torch.topk(att_cur_cat, tmp_topk, dim=0)[1]
                    in_top_p = h[in_top_p_idx]
                    in_p_targets = torch.full((tmp_topk, ), l, device=device).long()

                    logits = self.roi_clf(in_top_p)
                    inst_loss = self.loss_fn(logits, in_p_targets)
                    total_inst_loss += inst_loss
                

                h_weighted = torch.mm(att_embed,h) #n_classes,dim
                for c in range(self.num_classes):
                    slide_logits[b_i,c] = self.slide_clfs[c](h_weighted[c])
            if vis_mode == 3:
                return slide_logits,roi_attns
                
            return slide_logits,total_inst_loss
        else:
            if vis_mode==1:
                k, _, _ = x.shape
                x = self.vit(x)         # (n_rois_batch,dim) 
                return x
            if vis_mode==2:
                device = x.device
                # x --> n_rois_all.dim
                slide_logits = torch.empty(1, self.num_classes).float().to(device)
                h = x
                att_embed,_ = self.att_net(h) #k,n_classes

                roi_attns = att_embed
                att_embed = torch.transpose(att_embed,1,0) #n_classes,k
                att_embed = F.softmax(att_embed,dim=1) #n_classes,k

                h_weighted = torch.mm(att_embed,h) #n_classes,dim

                for c in range(self.num_classes):
                    slide_logits[0, c] = self.slide_clfs[c](h_weighted[c])
                
                return slide_logits,roi_attns

class SurvivalHead(nn.Module):
    """
    生存预测头，将特征映射到生存分析所需的格式
    """
    def __init__(self, dim, n_bins):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc = nn.Linear(dim, n_bins)
    
    def forward(self, x):
        x = self.norm(x)
        logits = self.fc(x)  # [batch, n_bins]
        hazards = torch.sigmoid(logits)  # 风险概率
        S = torch.cumprod(1 - hazards, dim=1)  # 生存概率
        return hazards, S


class ROAM_Survival_VIS(nn.Module):
    """
    ROAM_Survival的可视化版本
    支持生存分析的注意力热图生成，兼容原有可视化管道
    """
    def __init__(self, *, choose_num, num_patches, patch_dim, 
                 n_bins, roi_level, scale_type,
                 embed_weights, dim, depths, heads, mlp_dim, 
                 not_interscale=False, single_level=0,
                 dim_head=64, dropout=0., emb_dropout=0., attn_dropout=0., 
                 pool='cls', ape=True, attn_type='rel_sa', shared_pe=True):
        super().__init__()
        self.topk = choose_num
        self.n_bins = n_bins
        self.num_classes = n_bins  # 为了兼容接口
        self.roi_level = roi_level
        self.scale_type = scale_type
        self.embed_weights = embed_weights
        self.dim = dim

        # ViT backbone - 与原ROAM完全相同
        if self.scale_type == 'ms':
            if not_interscale:
                self.vit = PyramidViT_wo_interscale(num_patches=num_patches, embed_weights=self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
            else:
                if self.roi_level == 0:
                    self.vit = PyramidViT_0(num_patches=num_patches, embed_weights=self.embed_weights,
                                        patch_dim=patch_dim, dim=dim,
                                        depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                        pool=pool, dim_head=dim_head, dropout=dropout,
                                        emb_dropout=emb_dropout, ape=ape,
                                        attn_type=attn_type,
                                        shared_pe=shared_pe)
                elif self.roi_level == 1:
                    self.vit = PyramidViT_1(num_patches=num_patches, embed_weights=self.embed_weights,
                                        patch_dim=patch_dim, dim=dim,
                                        depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                        pool=pool, dim_head=dim_head, dropout=dropout,
                                        emb_dropout=emb_dropout, ape=ape,
                                        attn_type=attn_type,
                                        shared_pe=shared_pe)
                else:
                    self.vit = PyramidViT_2(num_patches=num_patches, embed_weights=self.embed_weights,
                                        patch_dim=patch_dim, dim=dim,
                                        depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                        pool=pool, dim_head=dim_head, dropout=dropout,
                                        emb_dropout=emb_dropout, ape=ape,
                                        attn_type=attn_type,
                                        shared_pe=shared_pe)
        else:
            self.vit = PyramidViT_SingleScale(num_patches=num_patches, embed_weights=self.embed_weights,
                                patch_dim=patch_dim, dim=dim,
                                depths=depths, level=single_level, heads=heads, mlp_dim=mlp_dim, 
                                pool=pool, dim_head=dim_head, dropout=dropout,
                                emb_dropout=emb_dropout, ape=ape,
                                attn_type=attn_type,
                                shared_pe=shared_pe)
        
        # 注意力网络 - 输出维度为1，用于全局注意力聚合
        self.att_net = Attention_net_gated(input_dim=dim, out_dim=dim//2, n_classes=1, dropout=attn_dropout)
        
        # ROI级分类器 - 用于实例级监督（如果需要）
        self.roi_clf = Classifier(dim, n_bins)
        
        # 生存预测头
        self.survival_head = SurvivalHead(dim, n_bins)
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, label=None, inst_level=False, vis=False, vis_mode=1):
        """
        前向传播，支持可视化模式
        
        Args:
            x: 输入特征
            label: 标签（用于实例级监督）
            inst_level: 是否使用实例级监督
            vis: 是否为可视化模式
            vis_mode: 可视化模式
                1: 返回ROI级特征
                2: 返回生存预测和注意力权重
                3: 端到端处理，返回生存预测和注意力权重
                
        Returns:
            根据vis_mode返回不同内容
        """
        if not vis:
            # 正常模式：兼容训练时的接口
            return self._forward_normal(x, label, inst_level)
        else:
            if vis_mode == 1:
                return self._forward_vis_mode_1(x)
            elif vis_mode == 2:
                return self._forward_vis_mode_2(x)
            elif vis_mode == 3:
                return self._forward_vis_mode_3(x)
            else:
                raise ValueError(f"Unsupported vis_mode: {vis_mode}")

    def _forward_normal(self, x_path, label=None, inst_level=False):
        """正常训练/推理模式"""
        device = x_path.device
        b, k, _, _ = x_path.shape
        
        # 处理特征
        x = rearrange(x_path, 'b k n d -> (b k) n d')
        x = self.vit(x)  # (b*k, dim)
        x = rearrange(x, '(b k) d -> b k d', b=b)  # (b, k, dim)
        
        # 初始化输出
        batch_hazards = torch.zeros(b, self.n_bins).float().to(device)
        batch_S = torch.zeros(b, self.n_bins).float().to(device)
        
        inst_loss_all = 0.0
        for b_i in range(b):
            h = x[b_i]  # (k, dim)
            
            # 计算注意力权重
            att_embed, _ = self.att_net(h)  # (k, 1)
            att_weights = F.softmax(att_embed.squeeze(-1), dim=0)  # (k,)
            
            # 实例级监督
            if inst_level and label is not None:
                tmp_topk = min(self.topk, k)
                l = label[b_i]
                if l < self.n_bins:
                    in_top_p_idx = torch.topk(att_weights, tmp_topk, dim=0)[1]
                    in_top_p = h[in_top_p_idx]
                    in_p_targets = torch.full((tmp_topk,), l, device=device).long()
                    
                    logits = self.roi_clf(in_top_p)
                    inst_loss = self.loss_fn(logits, in_p_targets)
                    inst_loss_all += inst_loss
            
            # 加权聚合
            h_weighted = torch.sum(att_weights.unsqueeze(-1) * h, dim=0)  # (dim,)
            
            # 生存预测
            hazards, S = self.survival_head(h_weighted.unsqueeze(0))
            batch_hazards[b_i] = hazards.squeeze(0)
            batch_S[b_i] = S.squeeze(0)
        
        router_weights = None
        return batch_hazards, batch_S, router_weights

    def _forward_vis_mode_1(self, x):
        """可视化模式1：返回ROI级特征"""
        k, _, _ = x.shape
        x = self.vit(x)  # (k, dim)
        return x

    def _forward_vis_mode_2(self, x):
        """可视化模式2：基于ROI特征计算注意力和预测"""
        device = x.device
        h = x  # (k, dim)
        
        # 计算注意力权重
        att_embed, _ = self.att_net(h)  # (k, 1)
        att_weights = F.softmax(att_embed.squeeze(-1), dim=0)  # (k,)
        
        # 加权聚合
        h_weighted = torch.sum(att_weights.unsqueeze(-1) * h, dim=0)  # (dim,)
        
        # 生存预测
        hazards, S = self.survival_head(h_weighted.unsqueeze(0))
        
        # 为了兼容原有接口，将生存风险转换为类似"分类概率"的格式
        # 使用负的累积生存概率作为"风险分数"
        risk_scores = -S  # (1, n_bins)
        
        # 返回格式：(slide_logits, roi_attns)
        # roi_attns需要是(k, n_classes)的格式以兼容原有代码
        roi_attns = att_weights.unsqueeze(-1).repeat(1, self.n_bins)  # (k, n_bins)
        
        return risk_scores, roi_attns

    def _forward_vis_mode_3(self, x):
        """
        可视化模式3：端到端处理
        
        智能处理不同维度的输入，统一转换为 PyramidViT 期望的格式：
        - 4D输入 [batch_size, num_ROIs, 84, dim]: 批量处理多个ROI
        - 3D输入 [num_ROIs, 84, dim]: 单个样本的多个ROI或单个ROI
        
        关键修复：保持 PyramidViT 模型内部期望的批次维度结构，
        避免在反向传播时出现维度不匹配的问题。
        """
        original_shape = x.shape
        print(f"Processing input with shape: {original_shape}")
        
        # 统一处理：将所有输入转换为标准的4D格式
        if x.dim() == 4:
            # 4D输入: [batch_size, num_ROIs, 84, dim]
            # 直接使用，这是PyramidViT期望的标准格式
            x_4d = x
            b, k = x.shape[0], x.shape[1]
            print(f"Using 4D input directly: batch_size={b}, num_ROIs={k}")
            
        elif x.dim() == 3:
            # 3D输入: [num_ROIs, 84, dim] 或 [1, 84, dim]
            # 关键修复：添加batch维度，转换为 [1, num_ROIs, 84, dim]
            x_4d = x.unsqueeze(0)  # [1, num_ROIs, 84, dim]
            b, k = 1, x.shape[0]
            print(f"Converted 3D to 4D: batch_size={b}, num_ROIs={k}")
            
        elif x.dim() == 2:
            # 2D输入: [84, dim] - 单个ROI的patch特征
            # 转换为 [1, 1, 84, dim] (1个batch，1个ROI)
            x_4d = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 84, dim]
            b, k = 1, 1
            print(f"Converted 2D to 4D: batch_size={b}, num_ROIs={k}")
            
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}. "
                           f"Expected 2D [84, dim], 3D [num_ROIs, 84, dim] or 4D [batch_size, num_ROIs, 84, dim], "
                           f"but got shape: {x.shape}")
        
        # 验证转换后的维度
        assert x_4d.dim() == 4, f"Internal error: expected 4D tensor, got {x_4d.dim()}D"
        assert x_4d.shape[2] == 84, f"Expected 84 patches, got {x_4d.shape[2]}"
        
        # 使用标准的PyramidViT前向传播路径
        # 这确保了与训练时完全一致的计算图结构
        print(f"Calling _forward_normal with shape: {x_4d.shape}")
        
        # 调用正常模式的前向传播，但只取出attention相关的部分
        device = x_4d.device
        
        # 重塑为PyramidViT期望的格式
        x_reshaped = rearrange(x_4d, 'b k n d -> (b k) n d')  # [(b*k), 84, dim]
        print(f"Reshaped for ViT: {x_reshaped.shape}")
        
        # 通过ViT骨干网络提取特征
        roi_features = self.vit(x_reshaped)  # [(b*k), dim]
        print(f"ViT output shape: {roi_features.shape}")
        
        # 重新整形回批次格式
        roi_features = rearrange(roi_features, '(b k) d -> b k d', b=b, k=k)  # [b, k, dim]
        print(f"Reshaped back to batch format: {roi_features.shape}")
        
        # 对于可视化，我们只处理第一个样本（batch_idx=0）
        h = roi_features[0]  # [k, dim] - 第一个样本的所有ROI特征
        print(f"Processing first sample with {h.shape[0]} ROIs")
        
        # 计算注意力权重
        att_embed, _ = self.att_net(h)  # [k, 1]
        att_weights = F.softmax(att_embed.squeeze(-1), dim=0)  # [k,]
        
        # 加权聚合
        h_weighted = torch.sum(att_weights.unsqueeze(-1) * h, dim=0)  # [dim,]
        
        # 生存预测
        hazards, S = self.survival_head(h_weighted.unsqueeze(0))  # 添加batch维度
        
        # 为了兼容原有接口，将生存风险转换为类似"分类概率"的格式
        # 使用负的累积生存概率作为"风险分数"
        risk_scores = -S  # [1, n_bins]
        
        # 返回格式：(slide_logits, roi_attns)
        # roi_attns需要是[k, n_classes]的格式以兼容原有代码
        roi_attns = att_weights.unsqueeze(-1).repeat(1, self.n_bins)  # [k, n_bins]
        
        print(f"Final output - risk_scores: {risk_scores.shape}, roi_attns: {roi_attns.shape}")
        return risk_scores, roi_attns


class ROAM_Survival(nn.Module):
    """
    ROAM模型的生存分析版本
    将原始的分类头替换为生存预测头，使其能够输出生存分析所需的格式
    """
    def __init__(self, num_patches, patch_dim, 
                 n_bins, roi_level, scale_type,  # n_bins替代num_classes
                 embed_weights, dim, depths, heads, mlp_dim, 
                 not_interscale=False, single_level=0,
                 dim_head=64, dropout=0., emb_dropout=0., attn_dropout=0., 
                 pool='cls', ape=True, attn_type='rel_sa', shared_pe=True,
                 choose_num=10): 
        super().__init__()
        self.n_bins = n_bins  # 生存分析的时间区间数
        self.num_classes = n_bins  # 为了兼容某些接口
        self.roi_level = roi_level
        self.scale_type = scale_type
        self.embed_weights = embed_weights
        self.topk = choose_num
        self.dim = dim

        # ViT backbone - 与原ROAM完全相同
        if self.scale_type == 'ms':
            if not_interscale:
                self.vit = PyramidViT_wo_interscale(num_patches=num_patches, embed_weights=self.embed_weights,
                                    patch_dim=patch_dim, dim=dim,
                                    depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                    pool=pool, dim_head=dim_head, dropout=dropout,
                                    emb_dropout=emb_dropout, ape=ape,
                                    attn_type=attn_type,
                                    shared_pe=shared_pe)
            else:
                if self.roi_level == 0:
                    self.vit = PyramidViT_0(num_patches=num_patches, embed_weights=self.embed_weights,
                                        patch_dim=patch_dim, dim=dim,
                                        depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                        pool=pool, dim_head=dim_head, dropout=dropout,
                                        emb_dropout=emb_dropout, ape=ape,
                                        attn_type=attn_type,
                                        shared_pe=shared_pe)
                elif self.roi_level == 1:
                    self.vit = PyramidViT_1(num_patches=num_patches, embed_weights=self.embed_weights,
                                        patch_dim=patch_dim, dim=dim,
                                        depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                        pool=pool, dim_head=dim_head, dropout=dropout,
                                        emb_dropout=emb_dropout, ape=ape,
                                        attn_type=attn_type,
                                        shared_pe=shared_pe)
                else:
                    self.vit = PyramidViT_2(num_patches=num_patches, embed_weights=self.embed_weights,
                                        patch_dim=patch_dim, dim=dim,
                                        depths=depths, heads=heads, mlp_dim=mlp_dim, 
                                        pool=pool, dim_head=dim_head, dropout=dropout,
                                        emb_dropout=emb_dropout, ape=ape,
                                        attn_type=attn_type,
                                        shared_pe=shared_pe)
        else:
            self.vit = PyramidViT_SingleScale(num_patches=num_patches, embed_weights=self.embed_weights,
                                patch_dim=patch_dim, dim=dim,
                                depths=depths, level=single_level, heads=heads, mlp_dim=mlp_dim, 
                                pool=pool, dim_head=dim_head, dropout=dropout,
                                emb_dropout=emb_dropout, ape=ape,
                                attn_type=attn_type,
                                shared_pe=shared_pe)
        
        # 注意力网络 - 输出维度为1，用于全局注意力聚合
        self.att_net = Attention_net_gated(input_dim=dim, out_dim=dim//2, n_classes=1, dropout=attn_dropout)
        
        # ROI级分类器 - 用于实例级监督（如果需要）
        self.roi_clf = Classifier(dim, n_bins)
        
        # 关键修改：使用单一的生存预测头，直接输出所有时间区间的风险概率
        self.survival_head = SurvivalHead(dim, n_bins)
        
        # 损失函数 - 这里不会用到，因为损失计算在训练引擎中
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 添加参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """
        ROAM_Survival模型参数初始化
        专注于关键的Linear层和预测头的初始化
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 对关键的预测层使用更精细的初始化
                if 'survival_head' in name or 'roi_clf' in name or 'classifier' in name:
                    # 预测层使用较小的初始化
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                else:
                    # 其他Linear层使用标准xavier初始化
                    nn.init.xavier_normal_(module.weight)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.Embedding):
                # 嵌入层使用normal分布初始化
                nn.init.normal_(module.weight, std=0.02)
                
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # BatchNorm层标准初始化
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def get_pathomics_features(self, hidden_states: torch.Tensor):
        """
        提供与Ciallo相同的接口，用于多模态融合
        """
        b, k, _, _ = hidden_states.shape
        hidden_states = rearrange(hidden_states, 'b k n d -> (b k) n d')
        features = self.vit(hidden_states)  # (batch_size * num_ROIs, dim)
        return features

    def forward(self, x_path, x_omics=None, label=None, inst_level=False):
        """
        前向传播，返回生存分析所需的格式
        
        Args:
            x_path: 病理图像特征 [batch_size, num_ROIs, num_patches, patch_dim]
            x_omics: 基因组特征（忽略，保持接口兼容性）
            label: 标签（用于实例级监督）
            inst_level: 是否使用实例级监督
            
        Returns:
            hazards: 风险概率 [batch_size, n_bins]
            S: 生存概率 [batch_size, n_bins]  
            router_weights: None（ROAM没有这个概念，为了兼容训练引擎）
        """
        device = x_path.device
        b, k, _, _ = x_path.shape
        
        # 向量化处理所有样本
        x = rearrange(x_path, 'b k n d -> (b k) n d')
        x = self.vit(x)  # (b*k, dim)
        x = rearrange(x, '(b k) d -> b k d', b=b)  # (b, k, dim)
        
        # 2. 初始化输出张量
        batch_hazards = torch.zeros(b, self.n_bins).float().to(device)
        batch_S = torch.zeros(b, self.n_bins).float().to(device)
        
        # 3. 对每个batch样本进行处理
        inst_loss_all = 0.0
        for b_i in range(b):
            h = x[b_i]  # (k, dim) - 当前样本的所有ROI特征
            
            # 4. 计算注意力权重 - 使用全局注意力
            att_embed, _ = self.att_net(h)  # (k, 1)
            att_weights = F.softmax(att_embed.squeeze(-1), dim=0)  # (k,)
            
            # 5. 实例级监督（可选）
            if inst_level and label is not None:
                tmp_topk = min(self.topk, k)
                l = label[b_i]
                if l < self.n_bins:  # 确保标签有效
                    # 选择注意力权重最高的ROI进行监督
                    in_top_p_idx = torch.topk(att_weights, tmp_topk, dim=0)[1]
                    in_top_p = h[in_top_p_idx]
                    in_p_targets = torch.full((tmp_topk,), l, device=device).long()
                    
                    logits = self.roi_clf(in_top_p)
                    inst_loss = self.loss_fn(logits, in_p_targets)
                    inst_loss_all += inst_loss
            
            # 6. 加权聚合ROI特征
            h_weighted = torch.sum(att_weights.unsqueeze(-1) * h, dim=0)  # (dim,)
            
            # 7. 通过生存预测头计算hazards和S
            hazards, S = self.survival_head(h_weighted.unsqueeze(0))  # 添加batch维度
            
            batch_hazards[b_i] = hazards.squeeze(0)  # 移除batch维度
            batch_S[b_i] = S.squeeze(0)  # 移除batch维度
        
        # 8. 返回符合训练引擎接口的格式
        router_weights = None  # ROAM没有这个概念
        return batch_hazards, batch_S, router_weights

