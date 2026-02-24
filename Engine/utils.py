import torch
import numpy as np
import torch.nn as nn
import pdb
import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import torch.nn.functional as F
import math
from itertools import islice
import collections
from torch.utils.data.dataloader import default_collate
# from models.model_porpoise import PorpoiseMMF, PorpoiseAMIL
from models.ROAM import ROAM,ROAM_Survival
from models.SNN import SNN
# from models.model_set_mil import MIL_Sum_FC_surv,MIL_Attention_FC_surv,MIL_Cluster_FC_surv
# from models.model_coattn import MCAT_Surv
from pack.prism import PRISM
from models.CMTA import CMTA
from models.SurMoE import SurMoE
from models.TransMIL import TransMIL
from models.MCAT import MCAT
from datasets.data_utils import set_worker_sharing_strategy
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import label_binarize
from torch import distributed as dist
from copy import deepcopy
from typing import Optional

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

# def build_model(args):
#     print(f'\nInit Model...{args.model_type}', end=' ')
#     if args.model_type == 'roam_survival':
#         model_dict = {
#             'n_bins': args.num_classes,
#             'num_patches': 84, 'patch_dim': 1024, 'roi_level': 0,
#             'scale_type': 'ms', 'single_level': 0, 'embed_weights': [0.3333, 0.3333, 0.3333],
#             'dim': 256, 'depths': [2,2,2,2,2], 'heads': 8, 'mlp_dim': 512,
#             'dim_head': 64, 'not_interscale': False, 'dropout': 0.2, 'emb_dropout': 0., 'attn_dropout': 0.25,
#             'pool': 'cls', 'ape': True, 'attn_type': 'rel_sa', 'shared_pe': True,
#             'choose_num': getattr(args, 'topk', 4),
#         }
#         model = ROAM_Survival(**model_dict)
#     elif args.model_type == 'ciallo':
#         model_dict = {'num_pathway':args.num_pathway, 'num_patches':84, 'patch_dim':1024, 
#                  'embed_weights':None, 'dim':256, 'depths':[2,2,2,1,1], 'heads':4, 'mlp_dim':512, 
#                  'roi_level':0, 'not_interscale':False, 'single_level':0,
#                  'scale_type':'ms','dim_head':64, 'dropout':0.1, 'emb_dropout':0., 'attn_dropout':0., 
#                  'pool':'cls', 'ape':True, 'attn_type':'rel_sa', 'shared_pe':True,
#                  'omic_sizes':args.omic_sizes,'num_classes':args.num_classes, 'fusion':args.fusion, 'model_size':"large",
#                  'task':args.task, 'max_rois':getattr(args, 'max_rois', 2000),  # 🚀 显存优化参数，支持grading任务
#                 #  # 🚀 新增LightMSF参数
#                 #  'use_lightmsf': getattr(args, 'use_lightmsf', True),  # 默认使用LightMSF
#                 #  'extract_scale': getattr(args, 'extract_scale', 'all'),  # 默认使用所有尺度
#                 #  'msf_config': getattr(args, 'msf_config', 'performance'),  # 使用新的增强配置
#                 #  # 🚀 新增MSCFusion参数
#                 #  'use_msc_fusion': getattr(args, 'use_msc_fusion', False),  # 默认不使用MSCFusion
#                  'msc_config': getattr(args, 'msc_config', 'default')}  # MSCFusion配置
#         model = Ciallo(**model_dict)
#     elif args.model_type == 'cmta':
#         # CMTA模型只支持前6个pathway，需要适配omic_sizes
#         cmta_omic_sizes = args.omic_sizes[:6] if len(args.omic_sizes) >= 6 else args.omic_sizes + [100] * (6 - len(args.omic_sizes))
#         model_dict = {
#             'omic_sizes': cmta_omic_sizes,
#             'num_classes': args.num_classes,
#             'fusion': args.fusion if args.fusion in ['concat', 'bilinear'] else 'concat',  # CMTA只支持concat和bilinear
#             'model_size': 'large',  # 使用large模型以获得更好的性能
#             'task': args.task,
#             'extract_scale': getattr(args, 'extract_scale', 'x20') , # 默认提取x20尺度特征
#             'max_rois':getattr(args, 'max_rois', 50000)
#         }
#         model = CMTA(**model_dict)
#     elif args.model_type == 'surmoe':
#         model_dict = {
#             'num_pathway': args.num_pathway, 'task':args.task,
#             'omic_sizes': args.omic_sizes,
#             'num_classes': args.num_classes,
#             'fusion': args.fusion if args.fusion in ['concat', 'bilinear'] else 'concat',
#             'model_size': 'small',
#             'extract_scale': getattr(args, 'extract_scale', 'x20') ,
#             'max_rois':getattr(args, 'max_rois', 2000),
#         }
#         model = SurMoE(**model_dict)
    
#     elif args.model_type == 'transmil':
#         model_dict = {
#             'num_classes': args.num_classes,
#             'task':args.task,
#             'extract_scale': getattr(args, 'extract_scale', 'x20'),
#             'max_rois':getattr(args, 'max_rois', 2000),
#         }
#         model = TransMIL(**model_dict)
#     elif args.model_type == 'mcat':
#         mcat_omic_sizes = args.omic_sizes[:6] if len(args.omic_sizes) >= 6 else args.omic_sizes + [100] * (6 - len(args.omic_sizes))
#         model_dict = {
#             'omic_sizes': mcat_omic_sizes,
#             'num_classes': args.num_classes,
#             'fusion': args.fusion if args.fusion in ['concat', 'bilinear'] else 'concat',
#             'model_size_wsi': 'small',
#             'model_size_omic': 'small',
#             'task':args.task,
#             'extract_scale': getattr(args, 'extract_scale', 'x20'),
#             'max_rois':getattr(args, 'max_rois', 2000),
#         }
#         model = MCAT(**model_dict)
#     elif args.model_type == 'snn':
#         # SNN模型配置：仅使用组学数据的神经网络
#         model_dict = {
#             'omic_sizes': args.omic_sizes,  
#             'num_classes': args.num_classes,
#             'model_size': 'large',  # 可选: 'small' 或 'big'
#             'num_pathway': args.num_pathway,
#             'task': args.task,
#         }
#         model = SNN(**model_dict)
#     else:
#         raise NotImplementedError
#     return model
def get_mil_model_params(args):
    genera_model_params = {
        "input_dim": args.input_dim,
        "num_classes": args.num_classes,
        "omic_sizes": args.omic_sizes,
        "use_batch_loss": args.use_batch_loss,
        # MultimodalBatchLoss hyperparameters
        "batch_loss_mem_size": getattr(args, 'batch_loss_mem_size', 4096),
        "batch_loss_shrink": getattr(args, 'batch_loss_shrink', 0.1),
        "batch_loss_mem_mem_weight": getattr(args, 'batch_loss_mem_mem_weight', 0.0),
        "batch_loss_use_ema": getattr(args, 'batch_loss_use_ema', False),
        "batch_loss_ema_decay": getattr(args, 'batch_loss_ema_decay', 0.99),
        "batch_loss_ema_weight": getattr(args, 'batch_loss_ema_weight', 1.0),
        "da_gated":args.da_gated,
        'num_pathway':args.num_pathway,
        "fusion": args.fusion,
        "model_size": 'small',
        "dropout": args.dropout,
        "act": args.act,
        "mil_norm": args.mil_norm,
        "cls_norm": args.cls_norm,
        "mil_bias": args.mil_bias,
        "inner_dim": args.inner_dim,
        "embed_feat": args.mil_feat_embed,
        'embed_feat_mlp_ratio': args.mil_feat_embed_mlp_ratio,
        'fc_norm_bn': True,
        'embed_norm_pos': args.embed_norm_pos,
        'feat_embed_type': args.mil_feat_embed_type,
        'pos': args.pos,
        
    }
    genera_trans_params = deepcopy(genera_model_params)
    genera_trans_params.update({
        'n_layers': args.n_layers,
        'pool': args.pool,
        'attn_dropout': args.attn_dropout,
        'deterministic': not args.no_determ,
        'sdpa_type': args.sdpa_type,
        'num_heads':args.num_heads,
        'fc_norm':True,
        'vit_norm': True,
        'attn_type': args.attn_type,
        'ffn_bias': True,
        'ffn_dp': 0.,
        'ffn_ratio': 4.,
    })

    return genera_model_params,genera_trans_params


def build_model(args,device):
	others = {}
	_,genera_trans_params = get_mil_model_params(args)
	if args.pack_bs:
		if args.mode =='M':
			model = PRISM(
            mil='msa',
            task = args.task,
            token_dropout=args.token_dropout,
            group_max_seq_len=args.pack_max_seq_len,
            min_seq_len=args.min_seq_len,
            pack_residual=not args.pack_no_residual,
            downsample_mode=args.pack_downsample_mode,
            downsample_type=args.pack_downsample_type,
            singlelabel=args.pack_singlelabel,
            residual_loss=args.pack_residual_loss,
            residual_downsample_r=args.pack_residual_downsample_r,
            pad_r=args.pack_pad_r,
            **genera_trans_params
            ).to(device)
	else:
		if args.model == 'roam_survival':
			model_dict = {
             'n_bins': args.num_classes,
             'num_patches': 84, 'patch_dim': 1024, 'roi_level': 0,
             'scale_type': 'ms', 'single_level': 0, 'embed_weights': [0.3333, 0.3333, 0.3333],
             'dim': 256, 'depths': [2,2,2,2,2], 'heads': 8, 'mlp_dim': 512,
             'dim_head': 64, 'not_interscale': False, 'dropout': 0.2, 'emb_dropout': 0., 'attn_dropout': 0.25,
             'pool': 'cls', 'ape': True, 'attn_type': 'rel_sa', 'shared_pe': True,
             'choose_num': getattr(args, 'topk', 4),
         }
			model = ROAM_Survival(**model_dict).to(device)
		elif args.model == 'cmta':
			model_dict = {
             'omic_sizes': args.omic_sizes,
			 'input_dim':getattr(args, 'input_dim', 1024),
             'num_classes': args.num_classes,
             'fusion': args.fusion if args.fusion in ['concat', 'bilinear'] else 'concat',  # CMTA只支持concat和bilinear
             'model_size': 'small',  
             'max_patches':getattr(args, 'max_patches', 3000),
         	}
			model = CMTA(**model_dict).to(device)
		elif args.model == 'surmoe':
			model_dict = {
             'num_pathway': args.num_pathway, 'task':args.task,
             'omic_sizes': args.omic_sizes,
             'num_classes': args.num_classes,
             'fusion': args.fusion if args.fusion in ['concat', 'bilinear'] else 'concat',
             'model_size': 'small',  
             'pack_bs': args.pack_bs,
        	}
			model = SurMoE(**model_dict).to(device)
    
		elif args.model == 'transmil':
			model_dict = {
             'input_dim':getattr(args, 'input_dim', 1024),
			 'n_classes':getattr(args, 'num_classes', 4),
             'dropout':getattr(args, 'dropout', 0.25),
			 'mil_norm':getattr(args, 'mil_norm', 'ln'),
			 'act':getattr(args, 'act', 'gelu')
         }
			model = TransMIL(**model_dict).to(device)
		elif args.model == 'mcat':
			mcat_omic_sizes = args.omic_sizes[:6] if len(args.omic_sizes) >= 6 else args.omic_sizes + [100] * (6 - len(args.omic_sizes))
			model_dict = {
             'omic_sizes': mcat_omic_sizes,
             'num_classes': args.num_classes,
             'fusion': args.fusion if args.fusion in ['concat', 'bilinear'] else 'concat',
             'model_size_wsi': 'small',
             'model_size_omic': 'small',
             'task':args.task,
             'extract_scale': getattr(args, 'extract_scale', 'x20'),
             'max_rois':getattr(args, 'max_rois', 2000),
         }
			model = MCAT(**model_dict).to(device)
		else:
			raise NotImplementedError
		
	return model,others

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, omic, label, event_time, c]

def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    cluster_ids = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
    omic = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    return [img, cluster_ids, omic, label, event_time, c]

class CollateMILSurvivalSig:
	def __init__(self, num_pathway,pack_bs):
		self.num_omics = num_pathway
		self.pack_bs = pack_bs

	def __call__(self, batch):
		# 预分配列表，减少动态分配
		path_features = []
		omic_list = [[] for _ in range(self.num_omics)]

		for item in batch:
			path_features.append(item['path_features'])
			for i in range(self.num_omics):
				omic_list[i].append(item['omic_list'][i])

		if self.pack_bs:
			path_features_output = path_features
		else:
			# 所有样本patch数量相同，使用stack创建3D张量
			path_features_output = torch.stack(path_features, dim=0)

		 # 批量处理omics数据
		omics_data = []
		batch_size = len(batch)
		for i in range(self.num_omics):
			if omic_list[i]:
				# 使用torch.stack而不是torch.cat，以支持批量训练
				# 每个omic_list[i][j]的形状为(num_genes,)，stack后变为(batch_size, num_genes)
				omics_data.append(torch.stack(omic_list[i], dim=0).type(torch.float32))
			else:
				# 如果没有数据，创建空的批次张量
				omics_data.append(torch.empty(batch_size, 0, dtype=torch.float32))
		targets = torch.LongTensor([item['target'] for item in batch])

		result = {
		'path_features': path_features_output,
		'omic_list': omics_data,
		'target': targets
	    }
		if any('event' in item for item in batch):
			result['event'] = torch.tensor([item['event'] for item in batch], dtype=torch.float32)
		if any('censorship' in item for item in batch):
			result['censorship'] = torch.tensor([item['censorship'] for item in batch],dtype=torch.float32)
		if any('case_id' in item for item in batch):
			result['case_id'] = [item['case_id'] for item in batch]
		return result

def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 2} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
    return loader 

def get_split_loader(split_dataset, args, training = False, weighted = False,):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': args.num_workers}
	if args.file_worker:
		kwargs['worker_init_fn'] = set_worker_sharing_strategy
	if args.pin_memory:
		kwargs['pin_memory']= True
	if args.num_workers > 0:
		kwargs['persistent_workers'] = True
		if args.prefetch:
			kwargs['prefetch_factor'] = args.prefetch_factor


	collate = CollateMILSurvivalSig(args.num_pathway,pack_bs=args.pack_bs)

	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(split_dataset)
			loader = DataLoader(split_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last,sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
		else:
			loader = DataLoader(split_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, collate_fn = collate, **kwargs)
	else:
		loader = DataLoader(split_dataset, batch_size=1, shuffle=False, collate_fn = collate, **kwargs)
	return loader


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
    seed = 7, label_frac = 1.0, custom_test_ids = None):
    indices = np.arange(samples).astype(int)
    
    pdb.set_trace()
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            remaining_ids = possible_indices

            if val_num[c] > 0:
                val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
                remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
                all_val_ids.extend(val_ids)

            if custom_test_ids is None and test_num[c] > 0: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            
            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)



def seed_torch(seed=2021):
		import random
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		# Set environment variable for deterministic CuBLAS operations
		os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
		#os.environ['PYTHONHASHSEED'] = str(seed)


class ModelEmaV3(nn.Module):
	def __init__(
			self,
			model,
			decay: float = 0.9999,
			min_decay: float = 0.0,
			update_after_step: int = 0,
			use_warmup: bool = False,
			warmup_gamma: float = 1.0,
			warmup_power: float = 2/3,
			device: Optional[torch.device] = None,
			foreach: bool = True,
			exclude_buffers: bool = False,
			mm_sche=None,
	):
		super().__init__()
		# make a copy of the model for accumulating moving average of weights
		self.module = deepcopy(model)
		self.module.eval()
		self.decay = decay
		self.min_decay = min_decay
		self.update_after_step = update_after_step
		self.mm_sche = mm_sche
		self.use_warmup = use_warmup
		self.warmup_gamma = warmup_gamma
		self.warmup_power = warmup_power
		self.foreach = foreach
		self.device = device  # perform ema on different device from model if set
		self.exclude_buffers = exclude_buffers
		if self.device is not None and device != next(model.parameters()).device:
			self.foreach = False  # cannot use foreach methods with different devices
			self.module.to(device=device)

	def get_decay(self, step: Optional[int] = None) -> float:
		"""
		Compute the decay factor for the exponential moving average.
		"""
		if step is None:
			return self.decay

		step = max(0, step - self.update_after_step - 1)
		# if step <= 0:
		# 	return 0.0
		if step < 0:
			return 0.0

		if self.use_warmup:
			decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
			decay = max(min(decay, self.decay), self.min_decay)
		else:
			decay = self.decay

		if self.mm_sche:
			decay = self.mm_sche[step]

		return decay

	@torch.no_grad()
	def update(self, model, step: Optional[int] = None):
		if self.decay == 1.:
			return None
		decay = self.get_decay(step) 
		if self.exclude_buffers:
			self.apply_update_no_buffers_(model, decay)
		else:
			self.apply_update_(model, decay)

	def apply_update_(self, model, decay: float):
		# interpolate parameters and buffers
		if self.foreach:
			ema_lerp_values = []
			model_lerp_values = []

			ema_state_dict = self.module.state_dict()
			model_state_dict = model.state_dict()
			for name, ema_v in ema_state_dict.items():
				if name in model_state_dict:
					model_v = model_state_dict[name]
				# ddp
				elif f"module.{name}" in model_state_dict:
					model_v = model_state_dict[f"module.{name}"]
				# torchcompile + ddp
				elif f"_orig_mod.module.{name}" in model_state_dict:
					model_v = model_state_dict[f"_orig_mod.module.{name}"]
				# torchcompile
				elif f"_orig_mod.{name}" in model_state_dict:
					model_v = model_state_dict[f"_orig_mod.{name}"]
				else:
					# print(f"Skipping parameter {name} as it's not found in source model")
					continue

				if ema_v.is_floating_point():
					ema_lerp_values.append(ema_v)
					model_lerp_values.append(model_v)
				else:
					ema_v.copy_(model_v)

			if hasattr(torch, '_foreach_lerp_'):
				torch._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1. - decay)
			else:
				torch._foreach_mul_(ema_lerp_values, scalar=decay)
				torch._foreach_add_(ema_lerp_values, model_lerp_values, alpha=1. - decay)
		else:
			for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
				if ema_v.is_floating_point():
					ema_v.lerp_(model_v.to(device=self.device), weight=1. - decay)
				else:
					ema_v.copy_(model_v.to(device=self.device))

	def apply_update_no_buffers_(self, model, decay: float):
		# interpolate parameters, copy buffers
		ema_params = tuple(self.module.parameters())
		model_params = tuple(model.parameters())
		if self.foreach:
			if hasattr(torch, '_foreach_lerp_'):
				torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
			else:
				torch._foreach_mul_(ema_params, scalar=decay)
				torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
		else:
			for ema_p, model_p in zip(ema_params, model_params):
				ema_p.lerp_(model_p.to(device=self.device), weight=1. - decay)

		for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
			ema_b.copy_(model_b.to(device=self.device))

	@torch.no_grad()
	def set(self, model):
		for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
			ema_v.copy_(model_v.to(device=self.device))

	def forward(self, *args, **kwargs):
		return self.module(*args, **kwargs)

def reduce_tensor(tensor, n):

	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= n
	return rt

def distributed_concat(tensor,num_sample):
	output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
	dist.all_gather(output_tensors, tensor)
	concat = torch.cat(output_tensors, dim=0)
	return concat[:num_sample]

def patch_shuffle(x,group=0,g_idx=None,return_g_idx=False):
	b,p,n = x.size()
	ps = torch.arange(p, device=x.device)

	# padding
	H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
	if group > H or group<= 0:
		return group_shuffle(x,group)
	_n = -H % group
	H, W = H+_n, W+_n
	add_length = H * W - p
	# print(add_length)
	ps = torch.cat([ps, torch.full((add_length,), -1, device=x.device, dtype=ps.dtype)])
	# patchify
	ps = ps.reshape(shape=(group,H//group,group,W//group))
	ps = torch.einsum('hpwq->hwpq',ps)
	ps = ps.reshape(shape=(group**2,H//group,W//group))
	# shuffle
	if g_idx is None:
		g_idx = torch.randperm(ps.size(0), device=x.device)
	ps = ps[g_idx]
	# unpatchify
	ps = ps.reshape(shape=(group,group,H//group,W//group))
	ps = torch.einsum('hwpq->hpwq',ps)
	ps = ps.reshape(shape=(H,W))
	idx = ps[ps>=0].view(p)
	
	if return_g_idx:
		return x[:,idx.long()],g_idx
	else:
		return x[:,idx.long()]

def group_shuffle(x,group=0):
	b,p,n = x.size()
	ps = torch.arange(p, device=x.device)
	if group > 0 and group < p:
		_pad = -p % group
		ps = torch.cat([ps, torch.full(( _pad,), -1, device=x.device, dtype=ps.dtype)])
		ps = ps.view(group,-1)
		g_idx = torch.randperm(ps.size(0), device=x.device)
		ps = ps[g_idx]
		idx = ps[ps>=0].view(p)
	else:
		idx = torch.randperm(p, device=x.device)
	return x[:,idx.long()]

def optimal_thresh(fpr, tpr, thresholds, p=0):
	loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
	idx = np.argmin(loss, axis=0)
	return fpr[idx], tpr[idx], thresholds[idx]

def save_cpk(args,model,random,train_loader,scheduler,optimizer,epoch,early_stopping,_metric_val,_te_metric,best_ckc_metric,best_ckc_metric_te,best_ckc_metric_te_tea,wandb):
	random_state = {
		'np': np.random.get_state(),
		'torch': torch.random.get_rng_state(),
		'py': random.getstate(),
		'loader': '',
	}
	ckp = {
		'model': model.state_dict(),
		'lr_sche': scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch+1,
		'k': args.fold_curr,
		'early_stop': early_stopping.state_dict(),
		'random': random_state,
		'ckc_metric': _metric_val+_te_metric,
		'val_best_metric': best_ckc_metric,
		'te_best_metric': best_ckc_metric_te+best_ckc_metric_te_tea,
		#'wandb_id': wandb.run.id if args.wandb else '',
	}
	torch.save(ckp, os.path.join(args.output_path, 'ckp.pt'))

def multi_class_scores(true_labels, pred_probs, classes):
	true_labels_bin = label_binarize(true_labels, classes=classes)

	macro_auc = roc_auc_score(true_labels_bin, pred_probs, average='macro', multi_class='ovr')

	predictions = np.argmax(pred_probs, axis=1)

	precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
	accuracy = accuracy_score(true_labels, predictions)

	return accuracy, macro_auc, precision, recall, fscore

def five_scores(bag_labels, bag_predictions,threshold_optimal=None):
	bag_predictions = bag_predictions[:,1]
	if threshold_optimal is None:
		fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
		fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
	# threshold_optimal=0.5
	auc_value = roc_auc_score(bag_labels, bag_predictions)
	this_class_label = np.array(bag_predictions)
	this_class_label[this_class_label>=threshold_optimal] = 1
	this_class_label[this_class_label<threshold_optimal] = 0
	bag_predictions = this_class_label
	precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
	accuracy = accuracy_score(bag_labels, bag_predictions)
	return accuracy, auc_value, precision, recall, fscore,threshold_optimal

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
	warmup_schedule = np.array([])
	warmup_iters = warmup_epochs * niter_per_ep
	if warmup_epochs > 0:
		warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

	iters = np.arange(epochs * niter_per_ep - warmup_iters)
	schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

	schedule = np.concatenate((warmup_schedule, schedule))
	assert len(schedule) == epochs * niter_per_ep
	return schedule

def update_best_metric(best_metric,val_metric):
	updated_best_metric = best_metric.copy()

	assert set(best_metric.keys()) == set(val_metric.keys()), "两个指标字典的键必须相同"

	for key in val_metric.keys():
		if key == "epoch":
			continue
		if key == "loss":
			if val_metric[key] < best_metric[key]:
				updated_best_metric[key] = val_metric[key]
		else:
			if val_metric[key] > best_metric[key]:
				updated_best_metric[key] = val_metric[key]

	return updated_best_metric

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=20, stop_epoch=50, verbose=False):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 20
			stop_epoch (int): Earliest epoch possible for stopping
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
		"""
		self.patience = patience
		self.stop_epoch = stop_epoch
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		try:
			self.val_loss_min = np.Inf
		except:
			self.val_loss_min = np.inf

	def __call__(self, args, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
		
		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, ckpt_name)
		elif score < self.best_score:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience and epoch > self.stop_epoch:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, ckpt_name)
			self.counter = 0

	def state_dict(self):
		return {
			'patience': self.patience,
			'stop_epoch': self.stop_epoch,
			'verbose': self.verbose,
			'counter': self.counter,
			'best_score': self.best_score,
			'early_stop': self.early_stop,
			'val_loss_min': self.val_loss_min
		}
	def load_state_dict(self,dict):
		self.patience = dict['patience']
		self.stop_epoch = dict['stop_epoch']
		self.verbose = dict['verbose']
		self.counter = dict['counter']
		self.best_score = dict['best_score']
		self.early_stop = dict['early_stop']
		self.val_loss_min = dict['val_loss_min']

	def save_checkpoint(self, val_loss, model, ckpt_name):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		#torch.save(model.state_dict(), ckpt_name)
		self.val_loss_min = val_loss
