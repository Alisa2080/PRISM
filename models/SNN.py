import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import SNN_Block,init_max_weights


class SNN(nn.Module):
    def __init__(self, omic_sizes=None, model_size: str='large', 
                 num_classes: int=4, task: str='survival', num_pathway: int=1):
        super(SNN, self).__init__()
        self.n_classes = num_classes
        self.num_pathway = num_pathway
        self.task = task
        self.size_dict = {'small': [256, 256], 'large': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict[model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)
        self.classifier = nn.Linear(hidden[-1], self.n_classes)
        init_max_weights(self)


    def forward(self, **kwargs):
        x_omic = [kwargs["x_omics"][i] for i in range(self.num_pathway)]
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0) #[1,n_gene, d]
        
        # 聚合多个pathway的特征：使用平均池化将[1, n_gene, d] -> [1, d]
        genomics_features = genomics_features.mean(dim=1)  # [1, d]
        logits = self.classifier(genomics_features)  # logits需要是[B x n_classes]的向量       
        
        # 根据任务类型返回不同格式的输出
        if self.task == "survival":
            # 生存任务：返回hazards, S, router_weights格式
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            # 为了与Engine兼容，router_weights设置为None
            router_weights = None
            return hazards, S, router_weights
        elif self.task in ["classification", "grading"]:
            # 分类和grading任务：返回logits, probs, router_weights格式
            probs = torch.softmax(logits, dim=1)
            # 为了与Engine兼容，router_weights设置为None
            router_weights = None
            return logits, probs, router_weights