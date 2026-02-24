import numpy as np

import torch
import torch.nn as nn

from models.utils import initialize_weights
from models.utils import BilinearFusion
from models.utils import SNN_Block
from models.utils import MultiheadAttention
import torch.nn.functional as F



class MoELayer(nn.Module):
    def __init__(self, num_pathway, dim=256):
        super().__init__()
        self.LinearFusion = LinearFusion(dim)
        self.AddFusion = AddFusion(dim, num_pathway)
        self.AttentionFusion = AttentionFusion(dim)
        self.IdentityFusion = IdentityFusion()
        self.num_experts = 4 
        self.routing_network = RoutingNetwork(self.num_experts, dim=dim)
        self.expert_list = [
            self.LinearFusion,
            self.AddFusion,
            self.AttentionFusion,
            self.IdentityFusion
        ]

    def forward(self, x1, x2, k=None):
        """
        x1, x2: two modalities, shape [b, n, dim]
        k: num of expert, default: None(soft mode)
        """
        logits = self.routing_network(x1, x2)  # [b, num_experts]
        bsz = x1.size(0)
        num_experts = self.num_experts

        if k is None or k >= num_experts:
            # soft mode, weighted sum for all experts 
            weights = torch.softmax(logits, dim=-1)  # [b, num_experts]
            out_gene = []
            out_img = []
            for expert in self.expert_list:
                out1, out2 = expert(x1, x2)  # [b, n, dim]
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
                out1, out2 = expert(x1[i:i+1], x2[i:i+1])  # [1, n, dim]
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
                out1, out2 = expert(x1, x2)  # [b, n, dim]
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

        return gene, img


class RoutingNetwork(nn.Module):
    """compute the routing logits for each expert"""
    def __init__(self, num_experts, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, num_experts)
        )

    def forward(self, x1, x2):
        # avg pooling for x1 and x2
        x1_mean = x1.mean(dim=1)  # [b, dim]
        x2_mean = x2.mean(dim=1)  # [b, dim]
        x = torch.cat([x1_mean, x2_mean], dim=-1)  # [b, dim * 2]
        logits = self.fc(x)  # [b, num_experts]
        return logits  # return logits, distributed in MoELayer


class LinearFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.AlphaDropout(p=0.25, inplace=False)) #nn.Linear(dim, dim)

    
    def forward(self, x1, x2):
        b, n1, dim = x1.size()
        _, n2, _ = x2.size()
        combined = torch.cat([x1, x2], dim=1)  # [b, n1+n2, dim]
        out = self.linear(combined)
        #print("---linear img",out[:, n1:, :].size())
        return out[:, :n1, :], out[:, n1:, :] # [b, n1, dim], [b, n2, dim]


class AddFusion(nn.Module):
    def __init__(self, dim,num_pathway):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(num_pathway, dim), nn.ELU(), nn.AlphaDropout(p=0.25, inplace=False))#nn.Linear(num_pathway, dim)
        self.linear2 = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.AlphaDropout(p=0.25, inplace=False)) #nn.Linear(dim, dim)
        self.linear3 = nn.Sequential(nn.Linear(dim, num_pathway), nn.ELU(), nn.AlphaDropout(p=0.25, inplace=False)) #nn.Linear(dim, num_pathway)
    def forward(self, x1, x2):

        _, n1, dim = x1.size()
        _, n2, _ = x2.size()

        if n1 < n2:  # x1_linear 的序列长度小于 x2_linear
            x1_linear = self.linear1(x1.permute(0, 2, 1))
            x2_linear = self.linear2(x2)
            x_out = x1_linear.permute(0, 2, 1) + x2_linear
            gene_out = x_out[:,:n1,:]
            img_out = x_out
        else:
            x1_linear = self.linear2(x1)
            x2_linear = self.linear3(x2.permute(0, 2, 1))
            x_out = x1_linear + x2_linear.permute(0, 2, 1)
            gene_out = x_out # [b, n1, dim]
            img_out = x_out[:,:n2,:] # [b, n2, dim]

        #print("------addfusion,img",img_out.size())
        #print("------addfusion,gene",gene_out.size())

        return gene_out, img_out

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


class SurMoE(nn.Module):
    def __init__(self,
    num_pathway,
    omic_sizes=[100, 200, 300, 400, 500, 600], 
    num_classes=4, fusion="concat", model_size="small",
    extract_scale="x20",task="survival",
    max_rois=2000):
        super(SurMoE, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = num_classes
        self.fusion = fusion
        self.max_rois = max_rois
        self.task = task
        self.extract_scale = extract_scale
        self.num_pathway = num_pathway
        # 定义多尺度特征的索引范围 (基于Ciallo数据集的84维结构)
        self.scale_indices = {
            "x20": (0, 64),    # 前64个patch: 20倍放大
            "x10": (64, 80),   # 中间16个patch: 10倍放大  
            "x5": (80, 84),    # 最后4个patch: 5倍放大
            "all": (0, 84)     # 全部84个patch
        }

        print(f"[SurMoE] 配置提取尺度: {extract_scale}")
        if extract_scale in self.scale_indices:
            start_idx, end_idx = self.scale_indices[extract_scale]
            print(f"[SurMoE] 将提取索引 {start_idx}-{end_idx-1} 的特征 (共{end_idx-start_idx}个patch)")

        ###
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        # Self-attention pooling for feature
        self.path_att_pooling = SelfAttentionPooling()
        self.gene_att_pooling = SelfAttentionPooling()

        self.clustering = ClusteringLayer(num_features=256, num_clusters=256)

        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        self.genomics_encoder = MoELayer(num_pathway, dim=256)
        self.genomics_decoder = MoELayer(num_pathway, dim=256)

        self.pathomics_encoder = MoELayer(num_pathway, dim=256)
        self.pathomics_decoder = MoELayer(num_pathway, dim=256)

        # P->G Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        self.G_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)

        # Classification Layer
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta genomics and pathomics features
        x_path = kwargs["x_path"]
        # 🎯 处理Ciallo风格的4D多尺度输入: (1, num_rois, 84, 1024)
        if len(x_path.shape) == 4:
            batch_size, num_rois, num_patches, feature_dim = x_path.shape
            
            # 验证是否为84维多尺度特征
            if num_patches == 84 and self.extract_scale in self.scale_indices:
                start_idx, end_idx = self.scale_indices[self.extract_scale]
                
                # 提取指定尺度的特征
                x_path_extracted = x_path[:, :, start_idx:end_idx, :]  # (1, num_rois, num_selected_patches, 1024)
                extracted_patches = end_idx - start_idx
                
                # 确保张量连续性，然后转换为2D格式: (1, num_rois * num_selected_patches, 1024)
                x_path_extracted = x_path_extracted.contiguous()
                x_path = x_path_extracted.reshape(batch_size, num_rois * extracted_patches, feature_dim)
            else:
                # 非84维特征或无效尺度配置，直接展平
                x_path = x_path.reshape(batch_size, num_rois * num_patches, feature_dim)
        
        x_omic = [kwargs["x_omics"][i] for i in range(self.num_pathway)] 

        # 🚀 显存优化：如果ROI数量过多，进行随机采样
        if x_path.shape[1] > self.max_rois:
            indices = torch.randperm(x_path.shape[1])[:self.max_rois]
            x_path = x_path[:, indices, : ]

        #------------------------------- embedding  -------------------------------#

        pathomics_features = self.pathomics_fc(x_path)
        pathomics_features = self.clustering(pathomics_features)  #[1,n_cluster,d]

        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0) #[1,n_gene, d]
        #------------------------------- encoder  -------------------------------#
        _, pathomics_features = self.pathomics_encoder(genomics_features, pathomics_features,k=4) #[1,n_gene, d]
        genomics_features, _ = self.genomics_encoder(genomics_features,pathomics_features,k=4) #[1,n_cluster,d]

        #------------------------------- cross-omics attention  -------------------------------#
        pathomics_in_genomics, path_att_score = self.P_in_G_Att( #[n_cluster,1,d]
            pathomics_features.transpose(0,1),
            genomics_features.transpose(0,1),
            genomics_features.transpose(0,1),
        )  
        genomics_in_pathomics, gene_att_score = self.G_in_P_Att( #[n_gene, 1, d]
            genomics_features.transpose(0,1),
            pathomics_features.transpose(0,1),
            pathomics_features.transpose(0,1),
        ) 

        pathomics_in_genomics = pathomics_in_genomics.transpose(0,1)
        genomics_in_pathomics = genomics_in_pathomics.transpose(0,1)

        #------------------------------- decoder  -------------------------------#
        _, pathomics_in_genomics = self.pathomics_decoder(genomics_in_pathomics, pathomics_in_genomics,k=4) #[1,n_gene, d]
        genomics_in_pathomics, _ = self.genomics_decoder(genomics_in_pathomics, pathomics_in_genomics,k=4) #[1,n_cluster,d]
        

        #------------------------------- fusion  -------------------------------#
        path_fusion = self.path_att_pooling(pathomics_in_genomics) #[1, dim]
        gene_fusion = self.gene_att_pooling(genomics_in_pathomics) #[1, dim]

        if self.fusion == "concat":
            fusion = self.mm(torch.concat((path_fusion,gene_fusion),dim=1))  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(gene_fusion, gene_fusion)  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        # predict
        logits = self.classifier(fusion)  # [1, n_classes]

        # 根据任务类型返回不同格式的输出
        if self.task == "survival":
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            # 为了与Engine兼容，返回router_weights (设置为None)
            router_weights = None
            return hazards, S, router_weights
        elif self.task in ["classification", "grading"]:
            # 分类和grading任务：返回logits和概率
            probs = torch.softmax(logits, dim=1)
            # 为了与Engine兼容，返回router_weights (设置为None)
            router_weights = None
            return logits, probs, router_weights
        else:
            raise NotImplementedError(f"Task [{self.task}] is not implemented")
