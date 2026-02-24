import torch
from torch import nn
import numpy as np
from timm.loss import AsymmetricLossSingleLabel

from pack.baseline import MILBase,MultimodalSAttention, SAttention
from pack.pack_util import *
from pack.packing import get_packs
from pack.ita import ITA
from pack.pack_loss import NLLSurvMulLoss, FocalLoss

class PRISM(nn.Module):
	def __init__(self,mil,task,token_dropout=0.5, group_max_seq_len=2048, min_seq_len=512,
				 downsample_mode='ads', residual_downsample_r=1,pad_r=False, singlelabel=False, 
				 pack_residual=True,residual_loss='bce',downsample_type='random', residual_ps_weight=False,
				 **mil_kwargs):
		super(PRISM, self).__init__()
		self.pool = mil_kwargs.get('pool', 'cls_token')
		mil_kwargs['attn_type'] = 'naive'
		self.num_classes = mil_kwargs.get('num_classes', 4)
		if mil == 'msa':
			self.mil = MILBase(aggregate_fn=MultimodalSAttention, **mil_kwargs)
			self.need_attn_mask = True		
		if pack_residual:
			if residual_loss == 'bce':
				self.residual_loss = nn.BCEWithLogitsLoss()
			elif residual_loss == 'asl_single':
				self.residual_loss = AsymmetricLossSingleLabel(gamma_pos=1, gamma_neg=4, eps=0.)
			elif residual_loss == 'focal':
				self.residual_loss = FocalLoss(alpha=0.25, gamma=2.0)
			elif residual_loss == 'ce':
				self.residual_loss = nn.CrossEntropyLoss()
			elif residual_loss == 'nll':
				self.residual_loss = NLLSurvMulLoss()
				
		self.downsample_mode = downsample_mode
		self.token_dropout = token_dropout
		self.group_max_seq_len = group_max_seq_len
		self.min_seq_len = min_seq_len
		self.no_norm_pad = mil_kwargs.get('mil_norm') == 'bn'
		self.downsample_r = residual_downsample_r
		self.residual = pack_residual
		self.residual_ps_weight = residual_ps_weight
		self.task = task
		
		if self.downsample_mode == 'ita':
			self.downsampler = ITA(r=self.downsample_r,
								   D=mil_kwargs.get('input_dim', 1024),
								   _type=downsample_type
								   )
			self.all_ds = True
		else:
			self.downsampler = None
			self.all_ds = False

		self.pad_r = pad_r
		self.singlelabel = singlelabel

		self.apply(initialize_weights)

	def apply_inference_downsample(self, x):
		if self.downsampler is not None:
			_tmp = self.downsampler.pool_factor
			_tmp_r = self.downsampler.r
			self.downsampler.pool_factor = None
			if _tmp is not None:
				self.downsampler.r = _tmp_r // _tmp
			x = self.downsampler(x, shuffle=False)
			self.downsampler.pool_factor = _tmp
			self.downsampler.r = _tmp_r

		return x

	def forward(self, x_path,omic_data,pos,label=None, loss=None,**mil_kwargs):
		if isinstance(x_path, list) and len(x_path) == 0:
			return None
		_pn = sum([len(_x_path) for _x_path in x_path])
		if self.training:
			B = len(x_path)

			if self.token_dropout > 0:
				_token_dropout = self.token_dropout

				max_feat_num = max([feat.size(0) for feat in x_path])
				keep_rate = 1 - self.token_dropout
				max_feat_num = int(int(max_feat_num * keep_rate) / self.downsample_r)
				if max_feat_num > self.group_max_seq_len:
					pack_len = min(self.group_max_seq_len * 2, self.group_max_seq_len * 4)
				else:
					pack_len = self.group_max_seq_len

				if self.task == 'survival':
					y, c = label
					_label = torch.cat([y.unsqueeze(-1), c.unsqueeze(-1)], dim=1)
				else:
					_label = label

				kept_dict, drop_dict = get_packs(
					x=x_path,
					token_dropout=_token_dropout,
					group_max_seq_len=pack_len,
					min_seq_len=self.min_seq_len,
					pool=self.pool,
					device=x_path[0].device,
					need_attn_mask=self.need_attn_mask,
					labels=_label,
					poses=pos,
					residual=self.residual,
					seq_downsampler=self.downsampler,
					enable_drop=self.residual,
					all_pu=self.all_ds,
					pad_r=self.pad_r,
				)

				x_path = kept_dict["patches"]
				attn_mask = kept_dict["attn_mask"]
				key_pad_mask = kept_dict["key_pad_mask"]
				num_feats = kept_dict["num_feats"]
				batched_feat_ids = kept_dict["batched_feat_ids"]
				cls_token_mask = kept_dict["cls_token_mask"]
				batched_feat_ids_1 = kept_dict["batched_feat_ids_1"]
				key_pad_mask_no_cls = kept_dict["key_pad_mask_no_cls"]
				pos = kept_dict["patch_positions"]
				batched_label = kept_dict["batched_labels"]
				batched_num_ps = kept_dict["batched_num_ps"]

				pack_args = {
					'attn_mask': attn_mask,
					'num_images': num_feats,
					'batched_image_ids': batched_feat_ids,
					'batched_image_ids_1': batched_feat_ids_1,
					'key_pad_mask': key_pad_mask,
					'key_pad_mask_no_cls': key_pad_mask_no_cls,
					'cls_token_mask': cls_token_mask,
					'no_norm_pad': self.no_norm_pad,
					'residual': False,
				}

				_kn = torch.sum(~key_pad_mask).item()
				total_elements = torch.numel(key_pad_mask)
				_pr = (1 - (_kn / total_elements)) * 100
				flattened_num_ps = [item for sublist in batched_num_ps for item in sublist]
				_kn_std = np.std(np.array(flattened_num_ps))
			else:
				x_path = torch.cat(x_path, dim=0).view(B, -1, self.mil.input_dim)
				pack_args = None
				_kn = _pn
				_pr = 0.
				_kn_std = 0.

			if self.residual:
				x_path_res = drop_dict['patches']
				label_res = drop_dict['batched_labels']
				key_pad_mask = drop_dict['key_pad_mask']
				key_pad_mask_res = drop_dict['key_pad_mask_res']
				key_pad_mask_no_cls_res = drop_dict['key_pad_mask_no_cls']
				batched_orig_indices = drop_dict['batched_orig_indices']
				batched_num_ps_res = drop_dict['batched_num_ps']

				if self.need_attn_mask:
					global_mask = ~key_pad_mask_res.cpu()
					global_attn_mask = (global_mask.unsqueeze(2) & global_mask.unsqueeze(1)).unsqueeze(1)
					global_attn_mask = global_attn_mask.to(x_path.device)
				else:
					global_attn_mask = None

				pack_res_args = {
					'attn_mask': global_attn_mask,
					'num_images': None,
					'batched_image_ids': None,
					'batched_image_ids_1': None,
					'key_pad_mask': key_pad_mask_res,
					'key_pad_mask_no_cls': key_pad_mask_no_cls_res,
					'cls_token_mask': None,
					'no_norm_pad': self.no_norm_pad,
					'residual': True,
					'batched_orig_indices':batched_orig_indices,
					'batched_num_ps': batched_num_ps_res,
				}
				_logits_res = self.mil(x_path_res, omic_data = omic_data,pack_args=pack_res_args, ban_norm=True, **mil_kwargs)
			
				if self.task == 'survival':
					_is_multi_lalbel = False
				else:
					_is_multi_lalbel = not (isinstance(self.residual_loss, nn.CrossEntropyLoss)
											or isinstance(self.residual_loss, AsymmetricLossSingleLabel))
				y_res = mixup_target_batched(
					label_res,
					num_classes=self.num_classes,
					multi_label=_is_multi_lalbel,
					batched_num_ps=batched_num_ps_res if self.residual_ps_weight else None,
					target_task=self.task
				)
				if isinstance(_logits_res, list):
					if self.task == 'survival':
						aux_loss = self.residual_loss(Y=y_res['Y'], c=y_res['c'], Y_censored=y_res['Y_censored'],
													  logits=_logits_res[0])
					else:
						if isinstance(self.residual_loss, AsymmetricLossSingleLabel) or self.singlelabel:
							y_res = y_res.argmax(dim=1)
						aux_loss = self.residual_loss(_logits_res[0], y_res)

				else:
					if self.task == 'survival':
						aux_loss = self.residual_loss(Y=y_res['Y'], c=y_res['c'], Y_censored=y_res['Y_censored'],
													  logits=_logits_res)

					else:
						if isinstance(self.residual_loss, AsymmetricLossSingleLabel) or self.singlelabel:
							y_res = y_res.argmax(dim=1)
						aux_loss = self.residual_loss(_logits_res, y_res)
			
			else:            
				aux_loss = 0.
			result = self.mil(x_path, omic_data = omic_data, pack_args=pack_args,pos=pos,**mil_kwargs)

			return result[0], aux_loss, result[1], _pn / B, _kn / B, _pr, _kn_std
		else:
			# 推理模式：处理列表输入并解包为张量
			if isinstance(x_path, list):
				# 如果是列表，合并所有张量（推理时通常batch_size=1，只有一个张量）
				if len(x_path) == 1:
					x_path_tensor = x_path[0].unsqueeze(0)  # 直接取出单个张量
				else:
					# 如果有多个张量，沿第0维拼接
				# 如果已经是张量，直接使用
					x_path_tensor = x_path.unsqueeze(0) # [1, N, D]
			x_res = self.apply_inference_downsample(x_path_tensor)
			_logits = self.mil(x_res, omic_data = omic_data,residual=True, **mil_kwargs)
			pack_args = None
			
		return _logits[0] # only return logits 
		