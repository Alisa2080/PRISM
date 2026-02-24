'''
Ref: https://github.com/zengwang430521/TCFormer/blob/1ea72a871b0932b51cf22334113a53c6a10d1f1a/tcformer_module/tcformer_utils.py#L384
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cluster_dpc_knn_gene_guided(token_dict, k=5, token_mask=None):
    """
    Assigns tokens in x to clusters in either x1 or x2 based on mean distance.

    Args:
        token_dict (dict): dict for token information.
        x1 (Tensor): first set of cluster centers [B, N1, C1].
        x2 (Tensor): second set of cluster centers [B, N2, C2].
        k (int): number of nearest neighbors used for local density.
        token_mask (Tensor[B, N]): mask indicating meaningful tokens.

    Returns:
        idx_cluster (Tensor[B, N]): cluster index of each token.
    """
    with torch.no_grad():
        x = token_dict['x']
        x1 = token_dict['omic1']
        x2 = token_dict['omic2']

        B, N, C = x.shape
        _, N1, C1 = x1.shape
        _, N2, C2 = x2.shape

        # Calculate pairwise distances
        dist_matrix1 = torch.cdist(x, x1)  # [B, N, N1]
        dist_matrix2 = torch.cdist(x, x2)  # [B, N, N2]

        # Compute mean distances
        mean_dist1 = dist_matrix1.mean(dim=-1)  # [B, N]
        mean_dist2 = dist_matrix2.mean(dim=-1)  # [B, N]

        # Assign clusters based on mean distance
        cluster_assignment = (mean_dist1 > mean_dist2).long()  # 0 for x1, 1 for x2

        # Assign tokens to the nearest center
        idx_cluster = cluster_assignment

    return idx_cluster

# cluster merge for hispathology patches.
def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        x = token_dict['x']
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        # 添加数值稳定性：限制指数参数避免溢出
        exp_arg = -(dist_nearest ** 2).mean(dim=-1)
        exp_arg = torch.clamp(exp_arg, min=-10, max=10)  # 防止exp()溢出或下溢
        density = exp_arg.exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new = agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged #[B, cluster_num, C]
    out_dict['token_num'] = cluster_num
    # out_dict['map_size'] = token_dict['map_size']
    # out_dict['init_grid_size'] = token_dict['init_grid_size']
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    return out_dict

# ClusterMergeNet block
class ClusterMergeNet(nn.Module):
    def __init__(self, sample_ratio, dim_out, target_clusters=None, knn_k=5):
        """
        ClusterMergeNet with DPC-KNN clustering and token merging.

        Compatibility additions for Ciallo:
        - Accept either a token dict (legacy) or a Tensor [B, N, C] directly.
        - If called with a Tensor, return only the clustered Tensor [B, K, C].
        - Allow fixed number of clusters via `target_clusters` or `num_clusters` in forward.

        Args:
            sample_ratio (float): ratio used to derive cluster count when no fixed K is given
            dim_out (int): feature dimension C
            target_clusters (int|None): default fixed cluster count K (e.g., num_pathway in Ciallo)
            knn_k (int): K for neighborhood in DPC-KNN
        """
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.target_clusters = target_clusters
        self.knn_k = knn_k
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)

    def _build_token_dict(self, x: torch.Tensor) -> dict:
        """Build a minimal token_dict required by clustering/merging from tensor x [B, N, C]."""
        B, N, _ = x.shape
        device = x.device
        token_dict = {
            'x': x,
            'idx_token': torch.arange(N, device=device).unsqueeze(0).expand(B, N),
            'agg_weight': torch.ones(B, N, 1, device=device),
        }
        return token_dict

    def forward(self, inputs, num_clusters: int = None):
        """
        Forward supports two input styles:
        - Tensor inputs: x [B, N, C] -> returns merged x [B, K, C]
        - Dict inputs (legacy): token_dict -> returns (down_dict, token_dict)

        For Ciallo, call with Tensor and optionally `num_clusters` (fixed K).
        If `num_clusters` is None, falls back to `self.target_clusters`, else to ceil(N*sample_ratio).
        """
        # Detect input type
        input_is_tensor = isinstance(inputs, torch.Tensor)

        # Prepare token_dict
        if input_is_tensor:
            if inputs.dim() != 3:
                raise RuntimeError(f"ClusterMergeNet expects [B, N, C] Tensor, got shape {tuple(inputs.shape)}")
            if inputs.size(-1) != self.dim_out:
                raise RuntimeError(f"Expected feature dim {self.dim_out}, got {inputs.size(-1)}")
            token_dict = self._build_token_dict(inputs)
        else:
            # keep legacy behavior but do not mutate caller's dict
            token_dict = inputs.copy()

        # Normalize and score tokens
        x = token_dict['x']  # [B, N, C]
        x = self.norm(x)
        token_score = self.score(x)  # [B, N, 1]
        token_weight = token_score.exp()  # [B, N, 1]
        token_dict['x'] = x
        token_dict['token_score'] = token_score

        B, N, C = x.shape

        # Determine cluster number
        K = num_clusters if num_clusters is not None else self.target_clusters
        if K is None:
            K = max(math.ceil(N * self.sample_ratio), 1)
        else:
            K = int(max(K, 1))
        # Ensure we do not request more clusters than tokens available
        K = min(K, N)
        
        # 安全检查：确保有足够的tokens进行聚类
        if N == 0:
            raise RuntimeError("ClusterMergeNet: received zero tokens (N=0)")
        if K > N:
            K = N  # 额外保险，虽然上面已经限制了

        # Run clustering and merge
        idx_cluster, cluster_num = cluster_dpc_knn(token_dict, K, k=self.knn_k)
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)

        # Return style depending on input type
        if input_is_tensor:
            return down_dict['x']  # [B, K, C]
        else:
            return down_dict, token_dict
        
        #  down_dict['x']:[B, cluster_num, C]
        # H, W = token_dict['map_size']
        # H = math.floor((H - 1) / 2 + 1) 
        # W = math.floor((W - 1) / 2 + 1)
        # down_dict['map_size'] = [H, W]
        

