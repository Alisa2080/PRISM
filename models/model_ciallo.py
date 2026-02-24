import torch
from torch import nn
import torch.nn.functional as F
from models.utils import MoE,MultiheadAttention,SNN_Block,SelfAttentionPooling,BilinearFusion, AttentionGatedFusion, SoftCluster,ClusteringLayer
from models.ROAM import ROAM
from models.MultiscaleFusion import LightMSF, create_light_msf
from models.MSCFusion import MSCFusion, create_msc_fusion
from models.DeformableAttention1D import DeformCrossAttention1D

def _compute_deform_attention_params(num_pathway, dim=256):
    """
    根据pathway数量自适应计算DeformCrossAttention1D的参数
    
    不同癌肿的pathway数量参考：
    - BLCA: 64 pathways
    - LUAD: ~188 pathways  
    - UCEC: ~284 pathways
    - GBMLGG: ~282 pathways
    
    Args:
        num_pathway: pathway数量
        dim: 特征维度 (默认256)
    
    Returns:
        dict: 包含优化参数的字典
    """
    # 基础参数设置
    params = {
        'dim': dim,
        'dim_head': 64,
        'dropout': 0.1
    }
    
    # 根据pathway数量动态调整参数
    if num_pathway <= 64:
        # 小规模: BLCA (64 pathways)
        params.update({
            'heads': 4,
            'downsample_factor': 4,
            'offset_scale': 1.0,
            'offset_groups': 4,
            'offset_kernel_size': 6
        })
    elif num_pathway <= 100:
        # 中小规模: 64-100 pathways
        params.update({
            'heads': 4,
            'downsample_factor': 6,
            'offset_scale': 1.2,
            'offset_groups': 4,
            'offset_kernel_size': 8
        })
    elif num_pathway <= 200:
        # 中等规模: LUAD等 (188 pathways)
        params.update({
            'heads': 4,
            'downsample_factor': 8,
            'offset_scale': 1.5,
            'offset_groups': 4,
            'offset_kernel_size': 10
        })
    elif num_pathway <= 300:
        # 大规模: UCEC等 (284 pathways)
        params.update({
            'heads': 4,
            'downsample_factor': 12,
            'offset_scale': 2.0,
            'offset_groups': 8,
            'offset_kernel_size': 14
        })
    else:
        # 超大规模: >=300 pathways
        params.update({
            'heads': 8,
            'downsample_factor': 16,
            'offset_scale': 2.5,
            'offset_groups': 8,
            'offset_kernel_size': 18
        })
    
    # 参数有效性检查与调整
    # 1. 确保offset_kernel_size >= downsample_factor
    if params['offset_kernel_size'] < params['downsample_factor']:
        params['offset_kernel_size'] = params['downsample_factor'] + 2
    
    # 2. 确保(offset_kernel_size - downsample_factor)为偶数
    diff = params['offset_kernel_size'] - params['downsample_factor']
    if diff % 2 != 0:
        params['offset_kernel_size'] += 1
    
    # 3. 确保heads能被offset_groups整除
    while params['heads'] % params['offset_groups'] != 0:
        if params['offset_groups'] > params['heads']:
            params['offset_groups'] = params['heads']
        else:
            params['heads'] += 1
    
    return params

class Ciallo(nn.Module):
    def __init__(self,num_pathway=64, num_patches=84, patch_dim=1024, 
                 embed_weights=[0.3333,0.3333,0.3333], dim=256, depths=[2,2,2,2,2], heads=8, mlp_dim=512, 
                 roi_level = 0, not_interscale=False, single_level=0,
                 scale_type='ms',dim_head=64, dropout=0., emb_dropout=0., attn_dropout=0., 
                 pool='cls', ape=True, attn_type='rel_sa', shared_pe=True,
                 omic_sizes=[100, 200, 300, 400, 500, 600],num_classes=4, fusion="concat", model_size="large",
                 task="survival", max_rois=2000, extract_scale="all", 
                 msc_config="default"):

        super(Ciallo, self).__init__()
        self.fusion = fusion
        self.num_classes = num_classes
        self.model_size = model_size
        self.omic_sizes = omic_sizes
        self.num_pathway = num_pathway
        self.dim = dim
        self.fusion = fusion
        self.task = task
        self.max_rois = max_rois  # 🚀 显存优化：限制最大ROI数量
        # self.use_lightmsf = use_lightmsf
        self.extract_scale = extract_scale
        # self.use_msc_fusion = use_msc_fusion
        
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256,128], "large": [1024,512,256]},
        }
        

        print(f"[Ciallo] 使用MSCFusion两尺度融合模块 (x20+x10)")
        print(f"[Ciallo] MSCFusion配置: {msc_config}")
            
            # 创建MSCFusion模块
        self.path_fusion = create_msc_fusion(
                config_name=msc_config,
                patch_dim=patch_dim,
                embed_dim=dim,
                dropout=dropout
            )
            
        # elif use_lightmsf:
        #     print(f"[Ciallo] 使用LightMSF多尺度融合模块")
        #     print(f"[Ciallo] LightMSF配置: {msf_config}, 提取尺度: {extract_scale}")
            
        #     # 创建LightMSF模块
        #     self.path_fusion = create_light_msf(
        #         config_name=msf_config,
        #         num_patches=num_patches,
        #         patch_dim=patch_dim,
        #         dim=dim,
        #         dropout=dropout,
        #         extract_scale=extract_scale,
        #         ape=ape
        #     )
            
        # else:
        #     print(f"[Ciallo] 使用传统ROAM多尺度融合模块")
        #     self.path_fusion = ROAM(num_patches = num_patches, patch_dim = patch_dim, roi_level = roi_level, scale_type = scale_type,
        #              embed_weights = embed_weights, dim = dim, depths = depths, heads = heads, mlp_dim = mlp_dim, 
        #              not_interscale = not_interscale, single_level = single_level, num_classes = num_classes,
        #              dim_head = dim_head, dropout = dropout, emb_dropout = emb_dropout, attn_dropout = attn_dropout, 
        #              pool = pool, ape = ape, attn_type = attn_type, shared_pe = shared_pe)
            

        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        self.genomics_encoder = MoE(num_pathway, dim=self.dim)
        self.genomics_decoder = MoE(num_pathway, dim=self.dim)
        self.pathomics_encoder = MoE(num_pathway, dim=self.dim)
        self.pathomics_decoder = MoE(num_pathway, dim=self.dim)

 
        self.P_in_G_Att = MultiheadAttention(embed_dim=self.dim, num_heads=1)
        self.G_in_P_Att = MultiheadAttention(embed_dim=self.dim, num_heads=1)


        self.path_att_pooling = SelfAttentionPooling(input_dim=self.dim)
        self.gene_att_pooling = SelfAttentionPooling(input_dim=self.dim)

        # Soft clustering to compress ROI tokens before MoE
        # self.path_cluster = SoftCluster(num_features=self.dim, num_clusters=self.num_pathway)
        self.path_cluster = ClusteringLayer(num_features=self.dim, num_clusters=self.num_pathway)

        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.GELU(), nn.Linear(hidden[-1], hidden[-1]), nn.GELU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        elif self.fusion == "attn_gate":
            self.mm = AttentionGatedFusion(in_dim=self.dim, out_dim=hidden[-1], gate_type="scalar")
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        # 根据任务类型创建不同的分类器
        if self.task == "survival":
            self.classifier = nn.Linear(hidden[-1], num_classes)
        elif self.task in ["classification", "grading"]:
            self.classifier = nn.Linear(hidden[-1], num_classes)
        else:
            raise NotImplementedError(f"Task [{self.task}] is not implemented")

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]  # (Batch_size= 1, num_ROI, 84, dim)
        x_omic = [kwargs["x_omics"][i] for i in range(self.num_pathway)] 

        # 🚀 显存优化：如果ROI数量过多，进行随机采样
        if x_path.shape[1] > self.max_rois:
            indices = torch.randperm(x_path.shape[1])[:self.max_rois]
            x_path = x_path[:, indices, :, :]

        #------------------------------- embedding  -------------------------------#
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0) #[1,n_gene, d]

        pathomics_features = self.path_fusion(x_path).unsqueeze(0) # (1, num_rois, dim)
        
        # compress ROIs to K=num_pathway prototypes via soft assignment
        pathomics_features = self.path_cluster(pathomics_features)
               
        pathomics_features, _, path_enc_weights = self.pathomics_encoder(pathomics_features, genomics_features,k=4) # [1,n_gene, dim]
        genomics_features, _, gene_enc_weights = self.genomics_encoder(genomics_features,pathomics_features,k=4) # [1, n_gene, dim]


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
        pathomics_in_genomics,_,path_dec_weights = self.pathomics_decoder(pathomics_in_genomics, genomics_in_pathomics,k=4) #[1,n_gene, d]
        genomics_in_pathomics,_, gene_dec_weights = self.genomics_decoder(genomics_in_pathomics, pathomics_in_genomics,k=4) #[1,n_gene,d]

        path_fusion = self.path_att_pooling(pathomics_in_genomics)
        gene_fusion = self.gene_att_pooling(genomics_in_pathomics)

        if self.fusion == "concat":
            fusion = self.mm(torch.concat((path_fusion,gene_fusion),dim=1))  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(path_fusion, gene_fusion)  # take cls token to make prediction
        elif self.fusion == "attn_gate":
            fusion = self.mm(path_fusion, gene_fusion)  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        # predict
        logits = self.classifier(fusion)  # [1, n_classes]

        router_weights = {
            "path_enc": path_enc_weights,
            "gene_enc": gene_enc_weights,
            "path_dec": path_dec_weights,
            "gene_dec": gene_dec_weights,
        }
        
        if self.task == "survival":
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, router_weights
        elif self.task in ["classification", "grading"]:
            # 分类和grading任务：返回logits和概率
            probs = torch.softmax(logits, dim=1)
            return logits, probs, router_weights
        else:
            raise NotImplementedError(f"Task [{self.task}] is not implemented")