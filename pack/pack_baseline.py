import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from timm.layers import use_fused_attn
from models.utils import default
from einops import repeat, rearrange
from functools import partial
from contextlib import suppress
from models.utils import exists
import numpy as np
from models.CMTA_util import SNN_Block, BilinearFusion
from nystrom_attention import NystromAttention
import math
import copy


class MultimodalBatchLoss(nn.Module):
    """
    Memory-augmented multimodal structural alignment loss.
    Implements cosine similarity + shrinkage with a lightweight cross-batch memory.

    Design goals:
    - Stabilize structure estimation under very small batch sizes (e.g., N=2–3)
    - Avoid gradient flow into memory entries (stored as buffers / detached)
    - Exclude (or downweight) memory-only interactions to prevent constant bias
    """
    def __init__(
        self,
        memory_size: int = 4096,
        shrink_lambda: float = 0.1,
        mem_mem_weight: float = 0.0,
        use_cosine: bool = True,
        eps: float = 1e-8,
        use_ema_targets: bool = False,
        ema_target_weight: float = 1.0,
    ):
        super(MultimodalBatchLoss, self).__init__()
        self.memory_size = int(memory_size)
        self.shrink_lambda = float(shrink_lambda)
        self.mem_mem_weight = float(mem_mem_weight)
        self.use_cosine = bool(use_cosine)
        self.eps = float(eps)
        self.use_ema_targets = bool(use_ema_targets)
        self.ema_target_weight = float(ema_target_weight)

        # Lazy buffers: allocated on first forward when dims are known
        self.register_buffer('mem_p', torch.empty(0))
        self.register_buffer('mem_g', torch.empty(0))
        self.register_buffer('mem_len', torch.zeros(1, dtype=torch.long))
        self.register_buffer('mem_ptr', torch.zeros(1, dtype=torch.long))

    def _ensure_buffers(self, device, dtype, dim_p: int, dim_g: int):
        if self.mem_p.numel() == 0:
            self.mem_p = torch.zeros(self.memory_size, dim_p, device=device, dtype=dtype)
            self.mem_g = torch.zeros(self.memory_size, dim_g, device=device, dtype=dtype)
            self.mem_len.zero_()
            self.mem_ptr.zero_()

    @torch.no_grad()
    def _enqueue(self, P: torch.Tensor, G: torch.Tensor):
        # P, G are detached before enqueue is called
        b = P.size(0)
        if b == 0:
            return
        ptr = int(self.mem_ptr.item())
        size = self.memory_size
        end = ptr + b
        if end <= size:
            self.mem_p[ptr:end].copy_(P)
            self.mem_g[ptr:end].copy_(G)
        else:
            first = size - ptr
            if first > 0:
                self.mem_p[ptr:].copy_(P[:first])
                self.mem_g[ptr:].copy_(G[:first])
            remain = b - first
            if remain > 0:
                self.mem_p[:remain].copy_(P[first:])
                self.mem_g[:remain].copy_(G[first:])
        new_len = min(self.memory_size, int(self.mem_len.item()) + b)
        self.mem_len[0] = new_len
        self.mem_ptr[0] = end % self.memory_size

    def _get_memory(self):
        m = int(self.mem_len.item())
        if m == 0:
            return None, None
        return self.mem_p[:m], self.mem_g[:m]

    @torch.no_grad()
    def reset(self):
        """Clear cross-batch memory buffers. Call at epoch start if desired."""
        if self.mem_p.numel() > 0:
            self.mem_p.zero_()
            self.mem_g.zero_()
        self.mem_len.zero_()
        self.mem_ptr.zero_()

    def _sim_matrix(self, X: torch.Tensor) -> torch.Tensor:
        # Cosine similarity by default
        if self.use_cosine:
            Xn = X / (X.norm(dim=1, keepdim=True) + self.eps)
            return Xn @ Xn.t()
        else:
            return X @ X.t()

    def _shrink(self, S: torch.Tensor, lam: float) -> torch.Tensor:
        if lam <= 0:
            return S
        N = S.size(0)
        return (1.0 - lam) * S + lam * torch.eye(N, device=S.device, dtype=S.dtype)

    def forward(
        self,
        pathomics_features: torch.Tensor,
        genomics_features: torch.Tensor,
        teacher_pathomics_features: torch.Tensor | None = None,
        teacher_genomics_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pathomics_features: [B, Dp] pathomics features
            genomics_features: [B, Dg] genomics features
        Returns:
            batch_loss: scalar loss for structural alignment (cosine + shrinkage)
        """
        assert pathomics_features.dim() == 2 and genomics_features.dim() == 2, "Inputs must be 2D [B, D]"

        B = pathomics_features.size(0)
        if B == 0:
            return pathomics_features.new_tensor(0.)

        P_cur = pathomics_features.view(B, -1)
        G_cur = genomics_features.view(B, -1)

        # lazily allocate buffers with current dims
        self._ensure_buffers(
            device=P_cur.device,
            dtype=P_cur.dtype,
            dim_p=P_cur.size(1),
            dim_g=G_cur.size(1),
        )

        # Gather memory (detached, no grad)
        P_mem, G_mem = self._get_memory()
        if P_mem is not None:
            P_all = torch.cat([P_cur, P_mem.detach()], dim=0)
            G_all = torch.cat([G_cur, G_mem.detach()], dim=0)
        else:
            P_all, G_all = P_cur, G_cur

        # Compute similarity matrices
        S_p = self._sim_matrix(P_all)
        S_g = self._sim_matrix(G_all)

        # Shrinkage to stabilize small-N estimation
        lam = self.shrink_lambda
        S_p = self._shrink(S_p, lam)
        S_g = self._shrink(S_g, lam)

        # Weighted MSE between similarity matrices
        N = S_p.size(0)
        device = S_p.device
        w = torch.zeros((N, N), device=device, dtype=S_p.dtype)
        # current-current and current-memory blocks weighted as 1.0
        b = B
        w[:b, :N] = 1.0
        w[:N, :b] = 1.0
        # memory-memory block (optional, default 0.0 to avoid constant bias)
        if N > b:
            w[b:, b:] = self.mem_mem_weight

        diff2 = (S_p - S_g) ** 2
        # Avoid zero division if w is all zeros (e.g., extreme corner)
        denom = w.sum().clamp_min(1.0)
        loss = (diff2 * w).sum() / denom

        # Optional: teacher targets (no memory for teacher, only current batch)
        if self.use_ema_targets and teacher_pathomics_features is not None and teacher_genomics_features is not None:
            Pt = teacher_pathomics_features.view(B, -1)
            Gt = teacher_genomics_features.view(B, -1)

            # student current-block similarity (no memory)
            S_p_cur = self._sim_matrix(P_cur)
            S_g_cur = self._sim_matrix(G_cur)
            S_p_cur = self._shrink(S_p_cur, self.shrink_lambda)
            S_g_cur = self._shrink(S_g_cur, self.shrink_lambda)

            # teacher similarity (no grad)
            with torch.no_grad():
                S_p_t = self._sim_matrix(Pt)
                S_g_t = self._sim_matrix(Gt)
                S_p_t = self._shrink(S_p_t, self.shrink_lambda)
                S_g_t = self._shrink(S_g_t, self.shrink_lambda)

            # Cross-modal alignment to teacher targets (symmetric)
            l_t = 0.5 * ((S_p_cur - S_g_t).pow(2).mean() + (S_g_cur - S_p_t).pow(2).mean())
            loss = loss + self.ema_target_weight * l_t

        # Update memory with current features (detach to stop grads)
        self._enqueue(P_cur.detach(), G_cur.detach())

        return loss

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

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

class AttentionPool(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim, bias=False)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            q,
            kv=None,
            mask=None,
            attn_mask=None
    ):
        q = self.norm(q)
        kv_input = default(kv, q)

        qkv = (self.to_q(q), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            deterministic: bool = True,
            sdpa_type: str = 'torch',
            residual = False,
            residual_conv_kernel = 33,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.deterministic = deterministic
        self.fused_attn_env = suppress
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(num_heads, num_heads, (kernel_size, 1), padding = (padding, 0), groups = num_heads, bias = False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sdpa_type = sdpa_type

    def forward(self, x: torch.Tensor, attn_mask=None, return_attn=False,freqs_cos=None,freqs_sin=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if freqs_cos is not None:
            # q_nocls,k_nocls = q[:,:,1:],k[:,:,1:]
            freqs_cos,freqs_sin = freqs_cos.type(q.dtype),freqs_sin.type(q.dtype)
            q = q * freqs_cos + rotate_half(q) * freqs_sin
            k = k * freqs_cos + rotate_half(k) * freqs_sin
            
            q = torch.cat([q[:,:,0].unsqueeze(-2),q], dim=2)
            k = torch.cat([k[:,:,0].unsqueeze(-2),k], dim=2)

        # flash-attn
        if return_attn:
            # B H N D
            q = q * self.scale
            # Only use CLS token query (first token)
            cls_q = q[:, :, 0:1, :]  # B H 1 D
            
            # Compute attention only for CLS token
            attn = cls_q @ k.transpose(-2, -1)  # B H 1 N
            
            if attn_mask is not None:
                # Apply mask only to CLS attention
                attn = attn.masked_fill(~attn_mask[:, :, 0:1, :], -torch.finfo(attn.dtype).max)
            
            attn = attn.softmax(dim=-1)  # B H 1 N
            attn = self.attn_drop(attn)
            # 这里为了节省显存，只返回cls token的attn，不对x做MSA
            # attn = q @ k.transpose(-2, -1)
            # if attn_mask is not None:
            #     attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)
            # attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            # x = attn @ v
            # x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # torch
            #with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            if self.sdpa_type == 'math' and self.training:
            #if return_attn:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                if attn_mask is not None:
                    attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v
                x = x.transpose(1, 2).reshape(B, N, C)
            elif attn_mask is not None or not self.sdpa_type == 'flash':
                # if self.sdpa_type == 'torch':
                # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                # q, k, v = qkv.unbind(0)
                # q, k = self.q_norm(q), self.k_norm(k)
                # if freqs_cos is not None:
                #     q[:,1:] = q[:,1:] * freqs_cos + rotate_half(q[:,1:]) * freqs_sin
                #     k[:,1:] = k[:,1:] * freqs_cos + rotate_half(k[:,1:]) * freqs_sin
                with self.fused_attn_env():
                    x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                x = x.transpose(1, 2).reshape(B, N, C)
            else:
                raise NotImplementedError

        # add depth-wise conv residual of values
        if self.residual:
            x = x + self.res_conv(v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            # cls token attn
            return x,attn[:,:,0,1:],v

        return x
    

# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


        
class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device,dtype=x.dtype)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W).contiguous()
        #cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x



class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, num_heads=8,dim=512, deterministic=True, attn_type='naive', mil_bias=False,dropout=0., sdpa_type='torch', norm=True,res=False):
        super().__init__()
        self.norm = norm_layer(dim, bias=mil_bias) if norm else nn.Identity()
        if attn_type == 'naive':
            self.attn = Attention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=dropout,
                deterministic=deterministic,
                sdpa_type=sdpa_type,
                residual=res,
            )
        elif attn_type == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head=dim//num_heads,
                heads = num_heads,
                num_landmarks = dim//2,    # number of landmarks
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=dropout
                )
        else:
            raise NotImplementedError

    def forward(self, x, attn_mask=None, need_attn=False, need_v=False, no_norm=False):
        if need_attn:
            z, attn, v = self.attn(self.norm(x), return_attn=need_attn, attn_mask=attn_mask)
            x = x + z
            if need_v:
                return x, attn, v
            else:
                return x, attn
        else:
            x = x + self.attn(self.norm(x), attn_mask=attn_mask)
            return x

class SAttention(nn.Module):
    def __init__(self, inner_dim=512, mil_bias=False, num_heads=8, n_layers=1, pos=None, pool='cls_token',attn_type='naive', attn_dropout=0., deterministic=True, sdpa_type='torch', fc_norm=True, vit_norm=True,attn_res=False,**kwargs):
        super(SAttention, self).__init__()
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=inner_dim)
        else:
            self.pos_embedding = nn.Identity()
        self.pos = pos
        if vit_norm:
            if fc_norm:
                self.norm = nn.Identity()
                self.fc_norm = nn.LayerNorm(inner_dim, bias=mil_bias)
            else:
                self.norm = nn.LayerNorm(inner_dim, bias=mil_bias)
                self.fc_norm = nn.Identity()
        else:
            self.norm = nn.Identity()
            self.fc_norm = nn.Identity()

        self.attn_type = attn_type

        self.layer1 = TransLayer(dim=inner_dim, num_heads=num_heads, attn_type=attn_type, mil_bias=mil_bias, dropout=attn_dropout,deterministic=deterministic, sdpa_type=sdpa_type, norm=vit_norm,res=attn_res)
        if n_layers >= 2:
            self.layers = [TransLayer(dim=inner_dim, num_heads=num_heads, attn_type=attn_type, mil_bias=mil_bias, dropout=attn_dropout,deterministic=deterministic, sdpa_type=sdpa_type, norm=vit_norm,res=attn_res) for i in range(n_layers - 1)]
            self.layers = nn.Sequential(*self.layers)
        else:
            self.layers = None

        self.cls_token = None
        self.pool = pool
        if pool == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, inner_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        elif pool == 'attn':
            self.attn_pool_queries = nn.Parameter(torch.randn(inner_dim))
            self.pool_fn = AttentionPool(dim=inner_dim, heads=8)
        else:
            raise NotImplementedError

    def insert_cls_token(self, x, cls_token_mask=None):
        num_cls_tokens = cls_token_mask.sum(dim=1)  # [batch]
        cls_tokens = repeat(self.cls_token, '1 n d -> (b n) d', b=num_cls_tokens.sum().item())

        new_x = torch.zeros(
            (x.shape[0], cls_token_mask.shape[1], x.shape[2]),
            dtype=x.dtype,
            device=x.device
        )

        cls_token_idx = 0
        for i in range(x.shape[0]):
            insert_positions = torch.where(cls_token_mask[i])[0]

            current_num_cls = num_cls_tokens[i].item()
            current_cls_tokens = cls_tokens[cls_token_idx:cls_token_idx + current_num_cls]
            cls_token_idx += current_num_cls

            src_pos = 0
            dst_pos = 0

            for k, insert_pos in enumerate(insert_positions):
                num_tokens_before = insert_pos - dst_pos

                if num_tokens_before > 0:
                    new_x[i, dst_pos:insert_pos] = x[i, src_pos:src_pos + num_tokens_before]
                    src_pos += num_tokens_before
                    dst_pos += num_tokens_before

                new_x[i, insert_pos] = current_cls_tokens[k]
                dst_pos += 1

            if src_pos < x.shape[1]:
                remaining = new_x.shape[1] - dst_pos
                if src_pos + remaining > x.shape[1]:
                    remaining = x.shape[1] - src_pos

                new_x[i, dst_pos:dst_pos + remaining] = x[i, src_pos:src_pos + remaining]

        return new_x

    def forward_ntrans(self,x, pack_args=None, return_attn=False, return_feat=False, pos=None,**kwargs):
        batch, num_patches, C = x.shape
        attn_mask = None
        cls_token_mask = None
        key_pad_mask = None
        if pack_args:
            attn_mask = default(pack_args['attn_mask'], None)
            num_images = pack_args['num_images']
            batched_image_ids = pack_args['batched_image_ids']
            key_pad_mask = pack_args['key_pad_mask']
            cls_token_mask = pack_args['cls_token_mask']
            batched_feat_ids_1 = pack_args['batched_image_ids_1']

        if self.cls_token is not None:
            # cls_token
            if cls_token_mask is None:
                cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
                x = torch.cat((cls_tokens, x), dim=1)
            else:
                x = self.insert_cls_token(x, cls_token_mask)

        attn = []
        # translayer1
        if return_attn:
            raise NotImplementedError

        if cls_token_mask is not None:
            for i in range(x.shape[0]):
                for k in range(num_images[i]):
                    _mask= batched_feat_ids_1[i] == k + 1
                    _mask_pos = _mask * (~cls_token_mask[i]).bool()
                    x[i, _mask] = self.layer1(x[i, _mask].unsqueeze(0)).squeeze(0)
                    x[i, _mask_pos] = self.pos_embedding(x[i, _mask_pos])
                    for _layer in self.layers:
                        x[i, _mask] = _layer(x[i, _mask].unsqueeze(0)).squeeze(0)
        else:
            x = self.layer1(x)
            if key_pad_mask is not None:
                x = x.masked_fill(key_pad_mask.unsqueeze(-1), 0.)
            x[:, 1:, :] = self.pos_embedding(x[:, 1:, :])
            if self.layers:
                for _layer in self.layers:
                    x = _layer(x)
                    if key_pad_mask is not None:
                        x = x.masked_fill(key_pad_mask.unsqueeze(-1), 0.)

        #---->cls_token
        x = self.norm(x)

        if cls_token_mask is None:
            if return_feat:
                x = self.fc_norm(x)
                return x[:, 0], x[:, 1:]
            return self.fc_norm(x[:, 0])
        else:
            return self.fc_norm(x[cls_token_mask])

    def forward(self, x, pack_args=None, return_attn=False, return_feat=False, pos=None,**kwargs):
        if self.attn_type == 'ntrans':
            return self.forward_ntrans(x, pack_args, return_attn, return_feat, pos)
        batch, num_patches, C = x.shape
        attn_mask = None
        cls_token_mask = None
        key_pad_mask = None
        if pack_args:
            attn_mask = default(pack_args['attn_mask'], None)
            num_images = pack_args['num_images']
            batched_image_ids = pack_args['batched_image_ids']
            key_pad_mask = pack_args['key_pad_mask']
            cls_token_mask = pack_args['cls_token_mask']
            batched_feat_ids_1 = pack_args['batched_image_ids_1']

        if self.cls_token is not None:
            # cls_token
            if cls_token_mask is None:
                cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
                x = torch.cat((cls_tokens, x), dim=1)
            else:
                x = self.insert_cls_token(x, cls_token_mask)
        attn = []
        # translayer1
        if return_attn:
            if self.attn_type == 'ntrans':
                raise NotImplementedError
            x, _attn, v = self.layer1(x, attn_mask=attn_mask, need_attn=True, need_v=True)
            attn.append(_attn.clone())
            
        else:
            if self.attn_type == 'ntrans':
                if cls_token_mask is not None:
                    for i in range(x.shape[0]):
                        for k in range(num_images[i]):
                            _mask= batched_feat_ids_1[i] == k + 1
                            x[i, _mask] = self.layer1(x[i, _mask].unsqueeze(0)).squeeze(0)
                else:
                    x = self.layer1(x)
                    if key_pad_mask is not None:
                        x = x.masked_fill(key_pad_mask.unsqueeze(-1), 0.)

            else:
                x = self.layer1(x, attn_mask=attn_mask)

        # add pos embedding
        if cls_token_mask is None:
            if self.pool == 'cls_token':
                x[:, 1:, :] = self.pos_embedding(x[:, 1:, :])
        else:
            for i in range(x.shape[0]):
                for k in range(num_images[i]):
                    #_mask_old = batched_feat_ids_1[i] == k + 1
                    _mask = ((batched_feat_ids_1[i] == k + 1) * (~cls_token_mask[i])).bool()
                    x[i, _mask] = self.pos_embedding(x[i, _mask])

        # translayer more
        if self.layers:
            for _layer in self.layers:
                if return_attn:
                    x, _attn, _ = _layer(x, attn_mask=attn_mask, need_attn=True, need_v=True)
                    attn.append(_attn.clone())
                else:
                    if self.attn_type == 'ntrans':
                        if cls_token_mask is not None:
                            for i in range(x.shape[0]):
                                for k in range(num_images[i]):
                                    _mask= batched_feat_ids_1[i] == k + 1
                                    x[i, _mask] = _layer(x[i, _mask].unsqueeze(0)).squeeze(0)
                        else:
                            x = _layer(x)
                            if key_pad_mask is not None:
                                x = x.masked_fill(key_pad_mask.unsqueeze(-1), 0.)
                    else:
                        x = _layer(x, attn_mask=attn_mask)

        # #---->cls_token
        x = self.norm(x)

        # -----> attn pool
        if self.pool == 'attn':
            if pack_args:
                arange = partial(torch.arange, device=x.device)
                max_queries = num_images.amax().item()

                queries = repeat(self.attn_pool_queries, 'd -> b n d', n=max_queries, b=x.shape[0])

                # attention pool mask

                image_id_arange = arange(max_queries)

                attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')

                attn_pool_mask = attn_pool_mask & rearrange(~key_pad_mask, 'b j -> b 1 j')

                attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')

                # attention pool

                x = self.pool_fn(queries, kv=x, attn_mask=attn_pool_mask) + queries

                x = rearrange(x, 'b n d -> (b n) d')

                # each batch element may not have same amount of images

                is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
                is_images = rearrange(is_images, 'b n -> (b n)')

                x = x[is_images]

            else:
                queries = repeat(self.attn_pool_queries, 'd -> b 1 d', b=x.shape[0])
                x = self.pool_fn(queries, kv=x) + queries

            return self.fc_norm(x)

        elif self.pool == 'cls_token':
            if cls_token_mask is None:
                if return_feat:
                    x = self.fc_norm(x)
                    return x[:, 0], x[:, 1:]
                return self.fc_norm(x[:, 0])
            else:
                return self.fc_norm(x[cls_token_mask])

class DAttention(nn.Module):
    def __init__(self, inner_dim=512, mil_bias=False, da_gated=False, cls_norm=None, fc_norm_bn=False,use_deterministic_softmax=False, mil_norm=None, **kwargs):
        super(DAttention, self).__init__()

        self.L = inner_dim  # 512
        self.D = 128  # 128
        self.K = 1
        self.da_gated = da_gated
        cls_norm = mil_norm if cls_norm is None else cls_norm
        if da_gated:
            self.attention_a = [
                nn.Linear(self.L, self.D, bias=mil_bias),
            ]
            self.attention_a += [nn.Tanh()]

            self.attention_b = [nn.Linear(self.L, self.D, bias=mil_bias),
                                nn.Sigmoid()]

            self.attention_a = nn.Sequential(*self.attention_a)
            self.attention_b = nn.Sequential(*self.attention_b)
            self.attention_c = nn.Linear(self.D, self.K, bias=mil_bias)
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.L, self.D, bias=mil_bias),
                nn.Tanh(),
                nn.Linear(self.D, self.K, bias=mil_bias)
            )

        if cls_norm == 'bn':
            if fc_norm_bn:
                self.norm1 = nn.BatchNorm1d(self.L * self.K)
            else:
                self.norm1 = nn.Identity()
        elif cls_norm == 'ln':
            self.norm1 = nn.LayerNorm(self.L * self.K, bias=mil_bias)
        else:
            self.norm1 = nn.Identity()

        self.use_deterministic_softmax = use_deterministic_softmax

    def forward(self, x, pack_args=None, return_attn=False, pos=None, ban_norm=False,**kwargs):
        if self.da_gated:
            A = self.attention_a(x)
            b = self.attention_b(x)
            A = A.mul(b)
            A = self.attention_c(A)
        else:
            A = self.attention(x)  # B N K
        A = torch.transpose(A, -1, -2)  # KxN
        if pack_args is not None:
            num_feats = pack_args['num_images']
            batched_feat_ids = pack_args['batched_image_ids']
            key_pad_mask = pack_args['key_pad_mask']

            if batched_feat_ids is not None:
                max_queries = num_feats.amax().item()
                arange = partial(torch.arange, device=x.device)
                # attention pool mask
                image_id_arange = arange(max_queries)
                attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_feat_ids, 'b j -> b 1 j')
                attn_pool_mask = attn_pool_mask & rearrange(~key_pad_mask, 'b j -> b 1 j')

                A = repeat(A, 'b 1 n -> b m n', m=max_queries)
                A = A.masked_fill(~attn_pool_mask, -torch.finfo(A.dtype).max)
            else:
                key_pad_mask = key_pad_mask.unsqueeze(1)
                A = A.masked_fill(key_pad_mask, -torch.finfo(A.dtype).max)

        A = F.softmax(A, dim=-1)

        x = torch.einsum('b k n, b n d -> b k d', A, x).squeeze(1)
        if pack_args is not None:
            if batched_feat_ids is not None:
                if len(x.shape) > 2:
                    x = rearrange(x, 'b n d -> (b n) d')
                # each batch element may not have same amount of images
                is_images = image_id_arange < rearrange(num_feats, 'b -> b 1')
                is_images = rearrange(is_images, 'b n -> (b n)')
                x = x[is_images]

        if pack_args is not None and ban_norm:
            pass
        else:
            x = self.norm1(x).squeeze(1)

        if return_attn:
            return x, A
        else:
            return x

class SNN(nn.Module):
    def __init__(self, inner_dim,omic_sizes,num_pathway,model_size,dropout,bias):
        super(SNN, self).__init__()
        self.omic_sizes = omic_sizes
        self.num_pathway = num_pathway
        self.dropout = dropout
        self.bias = bias
        self.size_dict = {
            "genomics": {
                "small": [inner_dim, inner_dim], 
                "large": [inner_dim, inner_dim, inner_dim, inner_dim]
            }
        }
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in self.omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0], bias=self.bias)]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=self.dropout, bias=self.bias))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)
    
    def forward(self, omic_data ,**kwargs):
        x_omic = [omic_data[i] for i in range(self.num_pathway)]
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).permute(1, 0, 2)
        return genomics_features

class XFuseLite(nn.Module):
    """
    XFuse-Lite: 低秩双路径跨门控融合模块
    输入:  F1, F2 形状 (B, D)
    输出:  F1', F2' 形状 (B, D)
    复杂度: O(B * D * r)
    """
    def __init__(self, dim: int, rank: int = None, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        # 推荐 r 取 D//8 或至少 16
        self.rank = max(16, dim // 8) if rank is None else rank

        # 预归一化（各模态独立）
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # —— Path-A: 低秩双线性 —— #
        self.U1 = nn.Linear(dim, self.rank, bias=False)
        self.U2 = nn.Linear(dim, self.rank, bias=False)
        self.V1 = nn.Linear(self.rank, dim, bias=False)
        self.V2 = nn.Linear(self.rank, dim, bias=False)

        # —— Path-B: 跨模态门控 + 低秩MLP —— #
        # 用对端生成门控
        self.g1_in  = nn.Linear(dim, self.rank, bias=True)
        self.g1_out = nn.Linear(self.rank, dim, bias=True)
        self.g2_in  = nn.Linear(dim, self.rank, bias=True)
        self.g2_out = nn.Linear(self.rank, dim, bias=True)
        # 调制后的低秩 MLP（先降后升，进一步细化）
        self.p1 = nn.Linear(dim, self.rank, bias=True)
        self.q1 = nn.Linear(self.rank, dim, bias=True)
        self.p2 = nn.Linear(dim, self.rank, bias=True)
        self.q2 = nn.Linear(self.rank, dim, bias=True)

        # 可学习残差缩放（ReZero 风格），初始化为0更稳
        self.gamma_1A = nn.Parameter(torch.zeros(1))
        self.gamma_1B = nn.Parameter(torch.zeros(1))
        self.gamma_2A = nn.Parameter(torch.zeros(1))
        self.gamma_2B = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # 小初始化保证初期近恒等
        for lin in [self.V1, self.V2, self.q1, self.q2, self.g1_out, self.g2_out]:
            nn.init.xavier_uniform_(lin.weight, gain=1e-1)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        for lin in [self.U1, self.U2, self.p1, self.p2, self.g1_in, self.g2_in]:
            nn.init.xavier_uniform_(lin.weight)
            if getattr(lin, "bias", None) is not None:
                nn.init.zeros_(lin.bias)

    def path_a(self, x1, x2):
        # 低秩双线性：m = gelu(U1 x1 ⊙ U2 x2)
        z1 = self.U1(x1)  # (B, r)
        z2 = self.U2(x2)  # (B, r)
        m = F.gelu(z1 * z2)  # (B, r)
        d1 = self.V1(m)      # (B, D)
        d2 = self.V2(m)      # (B, D)
        return d1, d2

    def path_b(self, x1, x2):
        # 跨模态门控
        g1 = torch.sigmoid(self.g1_out(F.gelu(self.g1_in(x2))))  # (B, D), 来自 x2
        g2 = torch.sigmoid(self.g2_out(F.gelu(self.g2_in(x1))))  # (B, D), 来自 x1
        # 门控调制 + 低秩MLP
        d1 = self.q1(F.gelu(self.p1(x1 * g1)))  # (B, D)
        d2 = self.q2(F.gelu(self.p2(x2 * g2)))  # (B, D)
        return d1, d2

    def forward(self, F1, F2):
        # Pre-Norm
        x1 = self.ln1(F1)
        x2 = self.ln2(F2)

        d1A, d2A = self.path_a(x1, x2)
        d1B, d2B = self.path_b(x1, x2)

        # 残差融合 + dropout + 可学习缩放
        F1p = F1 + self.dropout(self.gamma_1A * d1A + self.gamma_1B * d1B)
        F2p = F2 + self.dropout(self.gamma_2A * d2A + self.gamma_2B * d2B)

        return F1p, F2p
    

class MultimodalSAttention(nn.Module):   
    def __init__(self, 
                 inner_dim=256, 
                 num_heads=4,
                 mil_bias=False, 
                 n_layers=1, 
                 pos='ppeg', 
                 pool='cls_token',
                 attn_type='naive', 
                 attn_dropout=0., 
                 deterministic=True, 
                 use_batch_loss=False,
                 sdpa_type='torch', 
                 fc_norm=True, 
                 vit_norm=True,
                 attn_res=False,
                 # 新增多模态参数
                 omic_sizes=None,
                 num_pathway = 64,
                 num_classes=4,
                 fusion='concat',
                 model_size='small',
                 dropout=0.25,
                 **kwargs):
        super(MultimodalSAttention, self).__init__()
        
        self.use_batch_loss = use_batch_loss
        self.omic_sizes = omic_sizes or []
        self.num_classes = num_classes
        self.num_pathway = num_pathway
        self.fusion = fusion
        self.dropout = dropout
        self.pathomics_encoder = DAttention(inner_dim=inner_dim, mil_bias=mil_bias, **kwargs)
        self.genomics_encoder = SelfAttentionPooling(input_dim=inner_dim)
        
        self.genomics_fc = SNN(inner_dim,omic_sizes,num_pathway,model_size,dropout,mil_bias) 
        
        # EMA teacher toggles and modules (optional)
        self.use_ema_targets = bool(kwargs.pop('batch_loss_use_ema', False))
        self.ema_decay = float(kwargs.pop('batch_loss_ema_decay', 0.99))
        self._ema_weight_for_loss = float(kwargs.pop('batch_loss_ema_weight', 1.0))

        if self.use_ema_targets:
            self.pathomics_encoder_ema = copy.deepcopy(self.pathomics_encoder)
            self.genomics_fc_ema = copy.deepcopy(self.genomics_fc)
            self.genomics_encoder_ema = copy.deepcopy(self.genomics_encoder)

            for p in self.pathomics_encoder_ema.parameters():
                p.requires_grad = False
            for p in self.genomics_fc_ema.parameters():
                p.requires_grad = False
            for p in self.genomics_encoder_ema.parameters():
                p.requires_grad = False

            self.pathomics_encoder_ema.eval()
            self.genomics_fc_ema.eval()
            self.genomics_encoder_ema.eval()
        
        if self.use_batch_loss:
            # Initialize BatchLoss for structural alignment (with hyperparameters from kwargs or defaults)
            _mem_size = int(kwargs.pop('batch_loss_mem_size', 4096))
            _shrink = float(kwargs.pop('batch_loss_shrink', 0.1))
            _mmw = float(kwargs.pop('batch_loss_mem_mem_weight', 0.0))
            
            self.batch_loss_fn = MultimodalBatchLoss(
                memory_size=_mem_size,
                shrink_lambda=_shrink,
                mem_mem_weight=_mmw,
                use_ema_targets=self.use_ema_targets,
                ema_target_weight=self._ema_weight_for_loss,
            )
            
        self.fusion_layer1 = XFuseLite(dim=inner_dim,rank=128,dropout=dropout)
        self.fusion_layer2 = XFuseLite(dim=inner_dim,rank=128,dropout=dropout)
          
        self.classifier = nn.Linear(inner_dim * 2, self.num_classes)

    def forward(self, x_path, omic_data=None, pack_args=None, **kwargs):
        
        loss_batch = torch.tensor(0.).to(x_path.device)
        
        genomics_features = self.genomics_fc(omic_data)
        genomics_features = self.genomics_encoder(genomics_features)
        
        pathomics_features = self.pathomics_encoder(x_path, pack_args=pack_args, **kwargs)  # [B, inner_dim]
        
        if pack_args is not None and pack_args['residual']:
            pack_indices = pack_args['batched_orig_indices']  # 形如 List[List[int]] 或 Tensor[list]
            weights_pack = pack_args.get('batched_num_ps', None)  # 可选加权（每个包内各原始样本的token数）
        
            agg_list = []
            device = genomics_features.device
            dtype = genomics_features.dtype

            for i in range(len(pack_indices)):
                idx_i = pack_indices[i]
                if isinstance(idx_i, torch.Tensor):
                        idx_i = idx_i.to(device=device, dtype=torch.long)
                else:
                    idx_i = torch.tensor(idx_i, device=device, dtype=torch.long)

                feats_i = genomics_features.index_select(0, idx_i)  # [K_i, C]

                if weights_pack is not None:
                    w_i = weights_pack[i]
                    if isinstance(w_i, torch.Tensor):
                        w_i = w_i.to(device=device, dtype=dtype)
                    else:
                        w_i = torch.tensor(w_i, device=device, dtype=dtype)
                    if w_i.ndim == 1:
                        w_i = w_i.unsqueeze(-1)  # [K_i, 1]
                    w_sum = torch.clamp(w_i.sum(dim=0), min=1e-6)
                    feat_i = (feats_i * w_i).sum(dim=0) / w_sum  # 加权平均
                else:
                    feat_i = feats_i.mean(dim=0)

                agg_list.append(feat_i)

            genomics_features = torch.stack(agg_list, dim=0)  # [B_path, inner_dim]
                
        if pack_args is not None and not pack_args['residual'] and self.use_batch_loss:
            teacher_p = teacher_g = None
            if self.use_ema_targets and self.training:
                with torch.no_grad():
                    teacher_p = self.pathomics_encoder_ema(x_path, pack_args=pack_args, **kwargs)
                    g_ema = self.genomics_fc_ema(omic_data)
                    teacher_g = self.genomics_encoder_ema(g_ema)
            loss_batch = self.batch_loss_fn(
                pathomics_features,
                genomics_features,
                teacher_pathomics_features=teacher_p,
                teacher_genomics_features=teacher_g,
            )
              
        # 3. 多模态融合
        # if self.fusion == "concat":
        #         fusion_features = self.mm(torch.cat([pathomics_features, genomics_features], dim=1))
        # elif self.fusion == "bilinear":
        #         fusion_features = self.mm(pathomics_features, genomics_features)
        # else:
        #         raise NotImplementedError(f"Fusion [{self.fusion}] is not implemented")
        pathomics_features,genomics_features = self.fusion_layer1(pathomics_features,genomics_features)
        pathomics_features,genomics_features = self.fusion_layer2(pathomics_features,genomics_features)
        
        # 4. 分类
        logits = self.classifier(torch.cat([pathomics_features, genomics_features], dim=1))  # [B, num_classes]

        return [logits, loss_batch]

    @torch.no_grad()
    def update_ema_teachers(self):
        if not self.use_ema_targets:
            return
        m = self.ema_decay
        # pathomics encoder
        for p_t, p_s in zip(self.pathomics_encoder_ema.parameters(), self.pathomics_encoder.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
        # genomics fc
        for p_t, p_s in zip(self.genomics_fc_ema.parameters(), self.genomics_fc.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
        # genomics encoder
        for p_t, p_s in zip(self.genomics_encoder_ema.parameters(), self.genomics_encoder.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
    
def apply_function_nonpad(
        x: torch.Tensor,  # [B, L, in_dim]
        mask: torch.BoolTensor,  # [B, L]
        func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
    apply a function to non-padding elements of a tensor
    """
    B, L, D_in = x.shape

    x_nonpad = x[~mask]  # [N, D_in]

    x_nonpad_out = func(x_nonpad)  # [N, D_out]

    D_out = x_nonpad_out.shape[-1]
    x_out = torch.zeros(B, L, D_out, dtype=x_nonpad_out.dtype, device=x.device)
    x_out[~mask] = x_nonpad_out

    return x_out


class MILBase(nn.Module):
    def __init__(self, input_dim, dropout, act,aggregate_fn, mil_norm=None, mil_bias=False, inner_dim=256,
                 embed_feat=True, embed_norm_pos=0, **aggregate_args):
        super(MILBase, self).__init__()
        self.L = inner_dim  # 256
        self.K = 1
        self.embed = []
        self.mil_norm = mil_norm
        self.embed_norm_pos = embed_norm_pos
        self.input_dim = input_dim
        
        assert self.embed_norm_pos in (0, 1)

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
        elif mil_norm == 'ln':
            self.norm = nn.LayerNorm(input_dim, bias=mil_bias) if embed_norm_pos == 0 else nn.LayerNorm(inner_dim,bias=mil_bias)
        else:
            self.norm1 = self.norm = nn.Identity()

        if embed_feat:
            self.embed += [nn.Linear(input_dim, inner_dim, bias=mil_bias)]
            if act.lower() == 'gelu':
                self.embed += [nn.GELU()]
            else:
                self.embed += [nn.ReLU()]

            if dropout:
                self.embed += [nn.Dropout(0.25)]

        self.embed = nn.Sequential(*self.embed) if len(self.embed) > 0 else nn.Identity()
        
        self.aggregate = aggregate_fn(inner_dim=inner_dim, mil_bias=mil_bias, mil_norm=mil_norm, **aggregate_args)

    def forward_norm(self, x, pack_args, ban_bn=False):
        if self.mil_norm == 'bn' and not ban_bn:
            if pack_args is not None:
                key_pad_mask = pack_args['key_pad_mask_no_cls']
                if pack_args['no_norm_pad']:
                    x = apply_function_nonpad(x, key_pad_mask, self.norm)
                else:
                    x = torch.transpose(x, -1, -2)
                    x = self.norm(x)
                    x = torch.transpose(x, -1, -2)
            else:
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
        else:
            if pack_args is not None:
                key_pad_mask = pack_args['key_pad_mask_no_cls']
                if pack_args['no_norm_pad']:
                    x = apply_function_nonpad(x, key_pad_mask, self.norm)
                else:
                    x = self.norm(x)
            else:
                x = self.norm(x)
        return x

    def forward(self,x_path, omic_data,pack_args=None, ban_norm=False, ban_embed=False, residual=False, **mil_kwargs):
        if len(x_path.size()) == 2:
            x_path = x_path.unsqueeze(0)

        if not ban_embed:
            if self.embed_norm_pos == 0 and not ban_norm:
                x_path = self.forward_norm(x_path, pack_args)

            if pack_args is not None:
                x_path = apply_function_nonpad(x_path, pack_args['key_pad_mask_no_cls'], self.embed)
            else:
                x_path = self.embed(x_path)

            if self.embed_norm_pos == 1 and not ban_norm:
                x_path = self.forward_norm(x_path, pack_args)

        logits = self.aggregate(x_path,omic_data=omic_data,pack_args=pack_args,**mil_kwargs)

        return logits
