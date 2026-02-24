from __future__ import print_function

import os
import torch
import traceback
from timeit import default_timer as timer
import pandas as pd
import time
from einops._torch_specific import allow_ops_in_compiled_graph
from timm.utils import AverageMeter,distribute_bn
import wandb
import random
from collections import OrderedDict
import gc
import numpy as np
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from datasets.dataset_classification import Generic_WSI_Classification_Dataset, Generic_MIL_Classification_Dataset
from Engine.utils import seed_torch,update_best_metric,save_cpk,get_split_loader,build_model,ModelEmaV3
from Engine import build_engine
from Engine.train_utils import build_train
from Engine.options import _parse_args,more_about_config

def main(args,device):
	#### Create Results Directory
	seed_torch(args.seed)

	if args.task == "survival":
		cindex,cindex_std, te_cindex = [],[],[]
		cindex_ema, cindex_ema_std = [], []
		ckc_metric = [cindex,cindex_std]
		ckc_metric_ema = [cindex_ema,cindex_ema_std]
		te_ckc_metric = [te_cindex]
	else:
		acc, pre, rec,fs,auc,ck,acc_m,te_auc,te_fs=[],[],[],[],[],[],[],[],[]
		acc_std,auc_std,fs_std,ck_std,acc_m_std = [],[],[],[],[]
		acc_ema,pre_ema,rec_ema,fs_ema,auc_ema,acc_ema_std,auc_ema_std,fs_ema_std,ck_ema,acc_m_ema,ck_ema_std,acc_m_ema_std = [],[],[],[],[],[],[],[],[],[],[],[]
		ckc_metric = [auc,acc, pre, rec,fs,ck,acc_m,auc_std,acc_std,fs_std,ck_std,acc_m_std]
		ckc_metric_ema = [auc_ema,acc_ema, pre_ema, rec_ema,fs_ema,ck_ema,acc_m_ema,auc_ema_std,acc_ema_std,fs_ema_std,ck_ema_std,acc_m_ema_std]
		te_ckc_metric = [te_auc,te_fs]

	# Track a consistent label mapping across folds (for classification/grading)
	global_label_dict = None
	global_num_classes = None

	### Start 5-Fold CV Evaluation.
	for k in range(args.fold_start,args.num_splits):

		# build dataset
		if args.task == "survival":
			dataset = Generic_MIL_Survival_Dataset(
				csv_path = os.path.join("./csv",args.task,f"{args.cancer_type}_all_clean.csv"),
				gene_dir = args.gene_dir,
				cancer_type=args.cancer_type,
				data_dir=args.data_root_dir,
				shuffle=False,
				seed=args.seed,
				patient_strat=False,
				n_bins=4,
				label_col="survival_months",
				keep_same_psize=args.same_psize,
				same_psize_pad_type=args.same_psize_pad_type,
				min_seq_len=args.min_seq_len,
			)
		else:  # classification 或 grading
			dataset = Generic_MIL_Classification_Dataset(
				csv_path = os.path.join("./csv",args.task,f"{args.cancer_type}_all_clean.csv"),
				gene_dir = args.gene_dir,
				cancer_type=args.cancer_type,
				data_dir=args.data_root_dir,
				shuffle=False,
				seed=args.seed,
				patient_strat=False,
				label_col="oncotree_code",  # 使用oncotree_code作为分类标签			
				keep_same_psize=args.same_psize,
				same_psize_pad_type=args.same_psize_pad_type,
				min_seq_len=args.min_seq_len,
			)

		split_dir = os.path.join(args.data_root_dir,"splits", args.cancer_type,args.task)

		### Gets the Train + Val Dataset Loader.
		train_dataset, val_dataset = dataset.return_splits(from_id=False, 
				csv_path= os.path.join(split_dir, 'splits_{}.csv'.format(k)))

		# 处理分类/分级任务的全局标签映射，兼容 fold_start > 0 的情况
		if args.task in ["classification", "grading"]:
			if global_label_dict is None:
				# 第一次进入（可能是从非0折开始），记录全局映射
				global_label_dict = dataset.label_dict.copy()
				global_num_classes = len(global_label_dict)
				print(f"全局标签映射 (fold {k}): {global_label_dict}")
			else:
				# 检查后续fold的标签映射是否一致
				if dataset.label_dict != global_label_dict or len(dataset.label_dict) != global_num_classes:
					print(f"❌ 警告：fold {k} 的标签映射与首个fold不一致，这可能影响可比性")
					print(f"当前映射: {dataset.label_dict} (类别数: {len(dataset.label_dict)})")
					print(f"参考映射: {global_label_dict} (类别数: {global_num_classes})")
					# 此处仅警告，不强制改写以避免与已生成的 split 不一致。
		
		train_loader = get_split_loader(
            train_dataset,
            training=True,
            args=args,
        )
		
		val_loader = get_split_loader(
            val_dataset,
			training=False,
            args=args,
        )
		
		print(
            "training: {}, validation: {}".format(len(train_dataset), len(val_dataset))
        )
		
		# Optionally load class weights for this fold
		args.class_weights = None
		if getattr(args, 'use_class_weights', False):
			try:
				cw_file = os.path.join(split_dir, f"class_weights_fold_{k}.csv") if args.class_weights_from == 'fold' else os.path.join(split_dir, 'class_weights_overall.csv')
				if os.path.isfile(cw_file):
					cw_df = pd.read_csv(cw_file)
					if args.task == 'survival':
						# survival bins are 0..n_bins-1; map by 'class' index
						bins = int(cw_df['class'].max()) + 1 if 'class' in cw_df.columns else args.num_classes
						w = [1.0] * bins
						for _, row in cw_df.iterrows():
							cls_idx = int(row['class']) if 'class' in cw_df.columns else int(str(row['name']).split('_')[-1])
							if 0 <= cls_idx < bins:
								w[cls_idx] = float(row['weight'])
						args.class_weights = w
					else:
						# classification/grading: match by label string to dataset mapping
						w_map = {int(row['class']): float(row['weight']) for _, row in cw_df.iterrows()} if 'class' in cw_df.columns else {}
						name_map = {str(row['name']): float(row['weight']) for _, row in cw_df.iterrows()} if 'name' in cw_df.columns else {}
						num_cls = len(getattr(dataset, 'label_dict', {}))
						w = [1.0] * num_cls
						if hasattr(dataset, 'label_dict') and len(dataset.label_dict) > 0:
							for label_str, idx in dataset.label_dict.items():
								if label_str in name_map:
									w[idx] = float(name_map[label_str])
								elif idx in w_map:
									w[idx] = float(w_map[idx])
						args.class_weights = w
				else:
					print(f"[Info] Class weights file not found, skip: {cw_file}")
			except Exception as e:
				print(f"[Warn] Failed to load class weights: {e}")

		# build model, criterion, optimizer, schedular

		args.omic_sizes = train_dataset.omic_sizes	
		
		# 🚀 分类和grading任务：动态设置类别数
		if args.task in ["classification", "grading"]:
			args.num_classes = len(dataset.label_dict)
			print(f"{args.task}任务设置类别数: {args.num_classes}")
			print(f"标签映射: {dataset.label_dict}")
		
		args.fold_curr = k
		print('Start %d-fold cross validation: fold %d ' % (args.num_splits, k))
		
		ckc_metric, te_ckc_metric, ckc_metric_ema = one_fold(
			args, device, ckc_metric, te_ckc_metric, ckc_metric_ema, train_loader, val_loader, val_loader
		)

	if os.path.isfile(os.path.join(args.output_path, 'ckp.pt')):
		os.remove(os.path.join(args.output_path, 'ckp.pt'))

	if args.random_seed_fold > 0:
		if args.always_test:
			raise NotImplementedError('random_seed_fold and always_test are not supported in this time')
		if args.task == "survival":
			return cindex, cindex_std, cindex_ema
		else:
			return (
					acc, auc, fs, pre, rec, ck, acc_m, auc_std, acc_std, fs_std,
					acc_ema, auc_ema, fs_ema, pre_ema, rec_ema, ck_ema, acc_m_ema, auc_ema_std,
					acc_ema_std, fs_ema_std, acc_m_ema_std
				)
          
    
	if args.always_test:
		if args.wandb:
			if args.task == "survival":
				wandb.log({
                    "cross_val/te_cindex_mean":np.mean(np.array(te_cindex)),
                    "cross_val/te_cindex_std":np.std(np.array(te_cindex)),
                })
			else:
				wandb.log({
                    "cross_val/te_auc_mean":np.mean(np.array(te_auc)),
                    "cross_val/te_auc_std":np.std(np.array(te_auc)),
                    "cross_val/te_f1_mean":np.mean(np.array(te_fs)),
                    "cross_val/te_f1_std":np.std(np.array(te_fs)),
                })
    

	if args.wandb:
		if args.task == "survival":
			if len(cindex) > 0:
				wandb.log({
	                "cross_val/cindex_mean":np.mean(np.array(cindex)),
	                "cross_val/cindex_std_mean":np.mean(np.array(cindex_std)),
	                "cross_val/cindex_std":np.std(np.array(cindex)),
	            })
		else:
			if len(acc) > 0:
				wandb.log({
	                "cross_val/acc_mean":np.mean(np.array(acc)),
	                "cross_val/auc_mean":np.mean(np.array(auc)),
	                "cross_val/f1_mean":np.mean(np.array(fs)),
	                "cross_val/acc_std_mean":np.mean(np.array(acc_std)),
	                "cross_val/auc_std_mean":np.mean(np.array(auc_std)),
	                "cross_val/f1_std_mean":np.mean(np.array(fs_std)),
	                "cross_val/pre_mean":np.mean(np.array(pre)),
	                "cross_val/recall_mean":np.mean(np.array(rec)),
	                "cross_val/ck_mean":np.mean(np.array(ck)),
	                "cross_val/acc_micro_mean":np.mean(np.array(acc_m)),
	                "cross_val/acc_std":np.std(np.array(acc)),
	                "cross_val/auc_std":np.std(np.array(auc)),
	                "cross_val/f1_std":np.std(np.array(fs)),
	                "cross_val/pre_std":np.std(np.array(pre)),
	                "cross_val/recall_std":np.std(np.array(rec)),
	                "cross_val/ck_std":np.std(np.array(ck)),
	                "cross_val/acc_micro_std":np.std(np.array(acc_m)),
	            })
    
	if args.wandb:
		if args.task == "survival":
			if len(cindex_ema) > 0:
				wandb.log({
	                "cross_val/cindex_ema_mean":np.mean(np.array(cindex_ema)),
	                "cross_val/cindex_ema_std_mean":np.mean(np.array(cindex_ema_std)),
	                "cross_val/cindex_ema_std":np.std(np.array(cindex_ema)),
	            })
		else:
			if len(acc_ema) > 0:
				wandb.log({
	                "cross_val/acc_ema_mean":np.mean(np.array(acc_ema)),
	                "cross_val/auc_ema_mean":np.mean(np.array(auc_ema)),
	                "cross_val/f1_ema_mean":np.mean(np.array(fs_ema)),
	                "cross_val/ck_ema_mean":np.mean(np.array(ck_ema)),
	                "cross_val/acc_micro_ema_mean":np.mean(np.array(acc_m_ema)),
	                "cross_val/acc_ema_std_mean":np.mean(np.array(acc_ema_std)),
	                "cross_val/auc_ema_std_mean":np.mean(np.array(auc_ema_std)),
	                "cross_val/f1_ema_std_mean":np.mean(np.array(fs_ema_std)),
	                "cross_val/ck_ema_std_mean":np.mean(np.array(ck_ema_std)),
	                "cross_val/acc_micro_ema_std_mean":np.mean(np.array(acc_m_ema_std)),
	                "cross_val/pre_ema_mean":np.mean(np.array(pre_ema)),
	                "cross_val/recall_ema_mean":np.mean(np.array(rec_ema)),
	                "cross_val/acc_ema_std":np.std(np.array(acc_ema)),
	                "cross_val/auc_ema_std":np.std(np.array(auc_ema)),
	                "cross_val/f1_ema_std":np.std(np.array(fs_ema)),
	                "cross_val/pre_ema_std":np.std(np.array(pre_ema)),
	                "cross_val/recall_ema_std":np.std(np.array(rec_ema)),
	                "cross_val/ck_ema_std":np.std(np.array(ck_ema)),
	                "cross_val/acc_micro_ema_std":np.std(np.array(acc_m_ema)),
	            })

	if args.task == "survival":
		if len(cindex) > 0:
			print('Cross validation c-index mean: %.4f, std %.4f ' % (np.mean(np.array(cindex)), np.std(np.array(cindex))))
		else:
			print('Warning: No c-index values collected during cross validation')
		if args.test_type != 'main' or args.model_ema:
			if len(cindex_ema) > 0:
				print('Cross validation c-index EMA mean: %.4f, std %.4f ' % (np.mean(np.array(cindex_ema)), np.std(np.array(cindex_ema))))
			else:
				print('Warning: No EMA c-index values collected during cross validation')
	else:
            if len(acc) > 0:
                print('Cross validation accuracy mean: %.4f, std %.4f ' % (np.mean(np.array(acc)), np.std(np.array(acc))))
                print('Cross validation auc mean: %.4f, std %.4f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
                print('Cross validation precision mean: %.4f, std %.4f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
                print('Cross validation recall mean: %.4f, std %.4f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
                print('Cross validation fscore mean: %.4f, std %.4f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))
            else:
                print('Warning: No classification metrics collected during cross validation')

            if args.test_type != 'main' or args.model_ema:
                if len(acc_ema) > 0:
                    print('Cross validation accuracy EMA mean: %.4f, std %.4f ' % (np.mean(np.array(acc_ema)), np.std(np.array(acc_ema))))
                    print('Cross validation auc EMA mean: %.4f, std %.4f ' % (np.mean(np.array(auc_ema)), np.std(np.array(auc_ema))))
                else:
                    print('Warning: No EMA classification metrics collected during cross validation')

def one_fold(args,device,ckc_metric,te_ckc_metric,ckc_metric_ema,train_loader,val_loader,test_loader):
    # --->initiation

    torch.cuda.empty_cache()
    gc.collect()
    global stop_training
    
    loss_scaler = torch.amp.GradScaler(device=device,init_scale=2**args.amp_scale_index,growth_factor=2,growth_interval=args.amp_growth_interval,enabled=not args.amp_unscale)

    amp_autocast = torch.autocast

    # 2. 构建模型
    model,model_others = build_model(args,device)

    # 3. 构建 EMA (Exponential Moving Average) 模型
    if args.model_ema:
        model_ema = ModelEmaV3(model,decay=args.mm,use_warmup=args.mm_sche)
        if args.torchcompile:
            model_ema = torch.compile(model_ema)
    else:
        model_ema = None


    # 5. 使用 torch.compile 加速模型 (PyTorch 2.0+ 特性)
    if args.torchcompile:
        model = torch.compile(model,dynamic=True,mode=args.torchcompile_mode)
        allow_ops_in_compiled_graph()

    # --->build criterion,optimizer,scheduler,early-stopping
    criterion,optimizer,scheduler,early_stopping = build_train(args,model)

    # --->metric
    best_ckc_metric = [0. for i in range(len(ckc_metric))]
    best_ckc_metric_ema = [0. for i in range(len(ckc_metric))]
    best_ckc_metric_te = [0. for i in range(len(te_ckc_metric))]
    best_ckc_metric_te_ema = [0. for i in range(len(te_ckc_metric))]
    epoch_start,opt_thr,opt_main_ema = args.epoch_start,0,0
    _metric_val_ema = None

    # --->build engine
    train,validate = build_engine(args)

    # --->train loop
    train_time_meter = AverageMeter()
    if args.wandb and args.wandb_watch:
        wandb.watch(model,log_freq=int(args.log_iter / 10))
        print(f'Strart Watching Model')
    # if args.distributed:
    #     dist.barrier()
    try:
        if args.script_mode != 'test':
            for epoch in range(epoch_start, args.num_epoch):
                torch.cuda.empty_cache()
                gc.collect()
                train_loss,start,end = 0,0,0
                if not args.script_mode == "no_train":
                    # if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                        # train_loader.sampler.set_epoch(epoch)
                    if hasattr(train_loader.dataset, 'set_epoch'):
                        train_loader.dataset.set_epoch(epoch)

                    # 1. 训练一个 epoch
                    train_loss,start,end = train(args,model,model_ema,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,epoch,model_others)
                    train_time_meter.update(end-start)

                if args.script_mode == 'only_train':
                    if args.save_iter > 0 and epoch % args.save_iter == 0:
                        save_pt = {
                                    'model': model.state_dict(),
                                    'epoch': epoch
                                }
                        torch.save(save_pt, os.path.join(args.output_path, 'epoch_{epoch}_model.pt'.format(epoch=epoch)))

                    continue

                # 2. 在验证集上评估模型
                _metric_val,stop,test_loss, threshold_optimal,rowd_val = validate(args,model=model,loader=val_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,early_stopping=early_stopping,epoch=epoch,others=model_others)
                if args.test_type != 'main':
                    _metric_val,_metric_val_ema = _metric_val

                # 3. 如果有 EMA 模型，也进行评估
                if model_ema is not None:
                    if args.random_seed_fold > 0:
                        raise NotImplementedError('random_seed_fold and model_ema are not supported in this time')
                    _metric_val_ema,_, test_loss_ema,_,rowd_ema = validate(args,model=model_ema,loader=val_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,epoch=epoch,others=model_others)
                    if args.wandb:
                        rowd_ema = OrderedDict([ ('ema_'+_k,_v) for _k, _v in rowd_ema.items()])
                        if rowd_val is not None:
                            rowd_val.update(rowd_ema)

                # always run test_set in the training
                _te_metric = [0.,0.]
                if args.always_test:
                    _te_metric,_te_test_loss_log,rowd = validate(args,model=model,loader=test_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,status='test',others=model_others)
                    if args.wandb:
                        rowd = OrderedDict([ (str(args.fold_curr)+'-fold/te_'+_k,_v) for _k, _v in rowd.items()])
                        wandb.log(rowd)

                    if model_ema is not None:
                        _te_ema_metric,rowd,_te_ema_test_loss_log = validate(args,model=model_ema,loader=test_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,status='test')         
                        if args.wandb:
                            rowd = OrderedDict([ (str(args.fold_curr)+'-fold/te_ema_'+_k,_v) for _k, _v in rowd.items()])
                            wandb.log(rowd)

                    if args.test_type != 'main':
                        _te_metric,_te_ema_metric = _te_metric
                    else:
                        _te_ema_metric = None

                    if _te_metric[args.best_metric_index] > best_ckc_metric_te[args.best_metric_index]:
                        best_ckc_metric_te = [_te_metric[0],_te_metric[1]]
                        if args.wandb:
                            rowd = OrderedDict([
                                ("best_te_main",_te_metric[0]),
                                ("best_te_sub",_te_metric[1])
                            ])
                            rowd = OrderedDict([ (str(args.fold_curr)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                            wandb.log(rowd)
                    
                    if _te_ema_metric is not None:
                        if _te_ema_metric[args.best_metric_index] > best_ckc_metric_te_ema[args.best_metric_index]:
                            best_ckc_metric_te_ema = [_te_ema_metric[0],_te_ema_metric[1]]
                            if args.wandb:
                                rowd = OrderedDict([
                                    ("best_te_ema_main",_te_ema_metric[0]),
                                    ("best_te_ema_sub",_te_ema_metric[1])
                                ])
                                rowd = OrderedDict([ (str(args.fold_curr)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                                wandb.log(rowd)
                
                # logging and wandb
                if args.task == "survival":
                    print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, c-index: %.4f, time: %.4f(%.4f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[0], train_time_meter.val,train_time_meter.avg))
                else:
                    print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.4f, auc_value:%.4f, precision: %.4f, recall: %.4f, fscore: %.4f , time: %.4f(%.4f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[1], _metric_val[0], _metric_val[2], _metric_val[3], _metric_val[4], train_time_meter.val,train_time_meter.avg))
                rowd_val['epoch'] = epoch
                if args.wandb:
                    if args.random_seed_fold > 0:
                        rowd = OrderedDict([ (f"s{args.rsf_curr}-{args.fold_curr}-fold/val_{_k}",_v) for _k, _v in rowd_val.items()])
                    else:    
                        rowd = OrderedDict([ (str(args.fold_curr)+'-fold/val_'+_k,_v) for _k, _v in rowd_val.items()])
                    wandb.log(rowd)
                
                # update best metric
                if epoch == 0:
                    best_rowd = rowd_val
                else:
                    best_rowd = update_best_metric(best_rowd,rowd_val)

                # save the best model in the val_set
                if _metric_val[args.best_metric_index] > best_ckc_metric[args.best_metric_index]:
                    best_ckc_metric = _metric_val+[epoch]
                    best_rowd['epoch'] = epoch
                    opt_thr = threshold_optimal
                    if not os.path.exists(args.output_path):
                        os.mkdir(args.output_path)

                    _ema_ckp = model_ema.state_dict() if model_ema is not None else None
                    if 'model_ema' in model_others and model_others['model_ema'] is not None:
                        _ema_ckp = model_others['model_ema'].state_dict()
                    best_pt = {
                        'model': model.state_dict(),
                        'teacher': _ema_ckp,
                        'epoch': epoch
                    }
                    torch.save(best_pt, os.path.join(args.output_path, 'fold_{fold}_model_best.pt'.format(fold=args.fold_curr)))

                # save the best model_ema in the val_set
                if _metric_val_ema is not None:
                    if _metric_val_ema[args.best_metric_index] > best_ckc_metric_ema[args.best_metric_index]:
                        best_ckc_metric_ema = _metric_val_ema+[epoch]
                        if not os.path.exists(args.output_path):
                            os.mkdir(args.output_path)

                        _ema_ckp = model_ema.state_dict() if model_ema is not None else None
                        if 'model_ema' in model_others and model_others['model_ema'] is not None:
                            _ema_ckp = model_others['model_ema'].state_dict()
                        best_pt = {
                            'teacher': _ema_ckp,
                            'epoch': epoch
                        }
                        torch.save(best_pt, os.path.join(args.output_path, 'fold_{fold}_ema_model_best.pt'.format(fold=args.fold_curr)))


                if args.wandb:
                    if args.random_seed_fold > 0:
                        rowd = OrderedDict([ (f"s{args.rsf_curr}-{args.fold_curr}-fold/val_best_{_k}",_v) for _k, _v in best_rowd.items()])
                    else:
                        rowd = OrderedDict([ (str(args.fold_curr)+'-fold/val_best_'+_k,_v) for _k, _v in best_rowd.items()])
                    wandb.log(rowd)
                
                # save checkpoint
                save_cpk(args,model,random,train_loader,scheduler,optimizer,epoch,early_stopping,_metric_val,_te_metric,best_ckc_metric,best_ckc_metric_te,best_ckc_metric_te_ema,wandb)

                if args.save_iter > 0 and epoch % args.save_iter == 0:
                    save_pt = {
                                'model': model.state_dict(),
                                'epoch': epoch
                            }
                    torch.save(save_pt, os.path.join(args.output_path, 'epoch_{epoch}_model.pt'.format(epoch=epoch)))

                # if args.distributed:
                #     dist.barrier()

                if stop:
                    break

    except KeyboardInterrupt:
        pass
    
    best_std = None
    if os.path.exists(os.path.join(args.output_path, f'fold_{args.fold_curr}_model_best.pt')):
        best_std = torch.load(os.path.join(args.output_path, 'fold_{fold}_model_best.pt'.format(fold=args.fold_curr)),map_location='cpu',weights_only=True)

    if os.path.exists(os.path.join(args.output_path, f'fold_{args.fold_curr}_ema_model_best.pt')):
        best_std_ema = torch.load(os.path.join(args.output_path, 'fold_{fold}_ema_model_best.pt'.format(fold=args.fold_curr)),map_location='cpu',weights_only=True)

    if best_std is not None:
        info = model.load_state_dict(best_std['model'])
        print(f"Epoch {best_std['epoch']} Main Model Loaded: {info}")

    if model_ema is not None and best_std_ema['teacher'] is not None:
        info = model_ema.load_state_dict(best_std_ema['teacher'])
        print(f"Epoch {best_std_ema['epoch']} EMA Model Loaded: {info}")
    if args.test_type != 'main':
        info = model_others['model_ema'].load_state_dict(best_std_ema['teacher'])
        print(f"Epoch {best_std_ema['epoch']} EMA Model Loaded: {info}")

    metric_test,test_loss_log,rowd = validate(args,model=model,loader=test_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,status='test',others=model_others)

    if model_ema is not None:
        metric_test_ema,test_loss_log,rowd_ema = validate(args,model=model_ema,loader=test_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,status='test',others=model_others)
        if args.wandb:
            rowd_ema = OrderedDict([ ('ema_'+_k,_v) for _k, _v in rowd_ema.items()])
            rowd.update(rowd_ema)

    if args.wandb:
        if args.random_seed_fold > 0:
            rowd = OrderedDict([ (f"s{args.rsf_curr}/test_{_k}",_v) for _k, _v in rowd.items()])
        else:
            rowd = OrderedDict([ ('test_'+_k,_v) for _k, _v in rowd.items()])
        wandb.log(rowd)
    
    # if args.distributed:
    #     dist.barrier()

    if args.test_type != 'main':
        metric_test,metric_test_ema = metric_test
        [ckc_metric_ema[i].append(metric_test_ema[i]) for i,_ in enumerate(ckc_metric_ema)]
    elif model_ema is not None:
        [ckc_metric_ema[i].append(metric_test_ema[i]) for i,_ in enumerate(ckc_metric_ema)]
    
    # update metric for each-fold
    [ckc_metric[i].append(metric_test[i]) for i,_ in enumerate(ckc_metric)]

    if args.always_test:
        [te_ckc_metric[i].append(best_ckc_metric_te[i]) for i,_ in enumerate(te_ckc_metric)]
        
    return ckc_metric,te_ckc_metric,ckc_metric_ema

if __name__ == "__main__":
	args, args_text = _parse_args()
	args,device = more_about_config(args)
	
	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = True
		if args.no_determ:
            # unstable, but fast
			torch.backends.cudnn.benchmark = True
		else:
            #stable, always reproduce results
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
			if not args.no_deter_algo:
                # torch.use_deterministic_algorithms(True)
				torch.use_deterministic_algorithms(True, warn_only=True)
				torch.utils.deterministic.fill_uninitialized_memory=True
	if 'WANDB_MODE' in os.environ:
		if os.environ["WANDB_MODE"] == "offline":
			_output_wandb_cache = os.path.join(args.output_path,'wandb_cache',args.project,args.title)
 
	if not os.path.exists(os.path.join(args.output_path,args.project)):
		os.mkdir(os.path.join(args.output_path,args.project))
	args.output_path = os.path.join(args.output_path,args.project,args.title)
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)
	if args.wandb:
		_output = args.output_path
        
		wandb.init(project=args.project, name=args.title,config=args,dir=os.path.join(_output),tags=["unrecorded"])

        # Get the project root directory
		args.project_root = os.path.dirname(os.path.abspath(__file__))

		# Use os.path.join to create absolute paths
		wandb.save(os.path.join(args.project_root, 'main.py'),base_path=args.project_root, policy='now')
		wandb.save(os.path.join(args.project_root, 'options.py'),base_path=args.project_root, policy='now')
		wandb.save(os.path.join(args.project_root, 'engines/base_engine.py'),base_path=args.project_root, policy='now')
		wandb.save(os.path.join(args.project_root, 'modules/__init__.py'),base_path=args.project_root, policy='now')
		wandb.save(os.path.join(args.project_root, 'datasets/data_utils.py'),base_path=args.project_root, policy='now')
		wandb.save(os.path.join(args.project_root, 'modules/vit_mil.py'),base_path=args.project_root, policy='now')
        
		if args.config:
			for _config in args.config:
				wandb.save(os.path.join(args.project_root, f'config/{_config}'), base_path=args.project_root,policy='now')
  
		if args.pack_bs:
			wandb.save(os.path.join(args.project_root, 'modules/pack.py'), base_path=args.project_root,policy='now')
			wandb.save(os.path.join(args.project_root, 'modules/pack_baseline.py'), base_path=args.project_root,policy='now')
    
	localtime = time.asctime( time.localtime(time.time()) )

	print(localtime)
	try:
		if args.random_seed_fold > 0:
			_output = args.output_path
			if args.task == "survival":
				cindex_rsf,cindex_std_rsf,cindex_ema_rsf = [],[],[]
				for i in range(args.random_seed_fold):
					args.seed = args.seed_ori + i*100
					args.output_path = os.path.join(_output,f's{i}')
					args.rsf_curr = i
					print('Start %d-fold RSF validation: random %d ' % (args.random_seed_fold, i))
					cindex,cindex_std, cindex_ema = main(args=args,device=device)
					if args.wandb:
						wandb.log({
                            f"s{i}/cross_val/cindex_mean":np.mean(np.array(cindex)),
                            f"s{i}/cross_val/cindex_std_mean":np.mean(np.array(cindex_std)),
                            f"s{i}/cross_val/cindex_std":np.std(np.array(cindex)),
                        })
					cindex_rsf.append(np.mean(np.array(cindex)))
					cindex_std_rsf.append(np.mean(np.array(cindex_std)))
					cindex_ema_rsf.append(np.mean(np.array(cindex_ema)))
				if args.wandb:
					wandb.log({
                        "cross_val/cindex_mean":np.mean(np.array(cindex_rsf)),
                        "cross_val/cindex_std_mean":np.mean(np.array(cindex_std_rsf)),
                        "cross_val/cindex_std":np.std(np.array(cindex_rsf)),
                    })
                
				print('Cross validation c-index mean: %.4f, std %.4f ' % (np.mean(np.array(cindex_rsf)), np.std(np.array(cindex_rsf))))
			else:
				acc_rsf, auc_rsf, fs_rsf, pre_rsf, rec_rsf, ck_rsf, acc_m_rsf = [], [], [], [], [], [], []
				acc_std_rsf, auc_std_rsf, fs_std_rsf = [], [], []
				acc_ema_rsf, auc_ema_rsf, fs_ema_rsf, pre_ema_rsf, rec_ema_rsf, ck_ema_rsf, acc_m_ema_rsf = [], [], [], [], [], [], []
				acc_ema_std_rsf, auc_ema_std_rsf, fs_ema_std_rsf, acc_m_ema_std_rsf = [], [], [], []

				for i in range(args.random_seed_fold):
					args.seed = args.seed_ori + i * 100
					args.output_path = os.path.join(_output, f's{i}')
					args.rsf_curr = i
					print('Start %d-fold RSF validation: random %d ' % (args.random_seed_fold, i))

					(acc, auc, fs, pre, rec, ck, acc_m,
                     acc_std, auc_std, fs_std,
                     acc_ema, auc_ema, fs_ema, pre_ema, rec_ema, ck_ema, acc_m_ema,
                     acc_ema_std, auc_ema_std, fs_ema_std, acc_m_ema_std) = main(args=args, device=device)

					if args.wandb:
						if len(acc) > 0:  # 添加空列表保护
							wandb.log({
	                            f"s{i}/cross_val/acc_mean": np.mean(np.array(acc)),
	                            f"s{i}/cross_val/auc_mean": np.mean(np.array(auc)),
	                            f"s{i}/cross_val/f1_mean": np.mean(np.array(fs)),
	                            f"s{i}/cross_val/acc_std_mean": np.mean(np.array(acc_std)),
	                            f"s{i}/cross_val/auc_std_mean": np.mean(np.array(auc_std)),
	                            f"s{i}/cross_val/f1_std_mean": np.mean(np.array(fs_std)),
	                            f"s{i}/cross_val/pre_mean": np.mean(np.array(pre)),
	                            f"s{i}/cross_val/recall_mean": np.mean(np.array(rec)),
	                            f"s{i}/cross_val/ck_mean": np.mean(np.array(ck)),
	                            f"s{i}/cross_val/acc_micro_mean": np.mean(np.array(acc_m)),
	                            f"s{i}/cross_val/acc_std": np.std(np.array(acc)),
	                            f"s{i}/cross_val/auc_std": np.std(np.array(auc)),
	                            f"s{i}/cross_val/f1_std": np.std(np.array(fs)),
	                        })
						
						if len(acc_ema) > 0:  # 为EMA指标添加保护
							wandb.log({
	                            f"s{i}/cross_val/acc_ema_mean": np.mean(np.array(acc_ema)),
	                            f"s{i}/cross_val/auc_ema_mean": np.mean(np.array(auc_ema)),
	                            f"s{i}/cross_val/f1_ema_mean": np.mean(np.array(fs_ema)),
	                            f"s{i}/cross_val/pre_ema_mean": np.mean(np.array(pre_ema)),
	                            f"s{i}/cross_val/recall_ema_mean": np.mean(np.array(rec_ema)),
	                            f"s{i}/cross_val/ck_ema_mean": np.mean(np.array(ck_ema)),
	                            f"s{i}/cross_val/acc_micro_ema_mean": np.mean(np.array(acc_m_ema)),
	                            f"s{i}/cross_val/acc_ema_std_mean": np.mean(np.array(acc_ema_std)),
	                            f"s{i}/cross_val/auc_ema_std_mean": np.mean(np.array(auc_ema_std)),
	                            f"s{i}/cross_val/f1_ema_std_mean": np.mean(np.array(fs_ema_std)),
	                            f"s{i}/cross_val/acc_m_ema_std_mean": np.mean(np.array(acc_m_ema_std)),
	                            f"s{i}/cross_val/acc_ema_std": np.std(np.array(acc_ema)),
	                            f"s{i}/cross_val/auc_ema_std": np.std(np.array(auc_ema)),
	                            f"s{i}/cross_val/f1_ema_std": np.std(np.array(fs_ema)),
	                        })

					acc_rsf.append(np.mean(np.array(acc)))
					auc_rsf.append(np.mean(np.array(auc)))
					fs_rsf.append(np.mean(np.array(fs)))
					pre_rsf.append(np.mean(np.array(pre)))
					rec_rsf.append(np.mean(np.array(rec)))
					ck_rsf.append(np.mean(np.array(ck)))
					acc_m_rsf.append(np.mean(np.array(acc_m)))

					acc_std_rsf.append(np.mean(np.array(acc_std)))
					auc_std_rsf.append(np.mean(np.array(auc_std)))
					fs_std_rsf.append(np.mean(np.array(fs_std)))

					acc_ema_rsf.append(np.mean(np.array(acc_ema)))
					auc_ema_rsf.append(np.mean(np.array(auc_ema)))
					fs_ema_rsf.append(np.mean(np.array(fs_ema)))
					pre_ema_rsf.append(np.mean(np.array(pre_ema)))
					rec_ema_rsf.append(np.mean(np.array(rec_ema)))
					ck_ema_rsf.append(np.mean(np.array(ck_ema)))
					acc_m_ema_rsf.append(np.mean(np.array(acc_m_ema)))

					acc_ema_std_rsf.append(np.mean(np.array(acc_ema_std)))
					auc_ema_std_rsf.append(np.mean(np.array(auc_ema_std)))
					fs_ema_std_rsf.append(np.mean(np.array(fs_ema_std)))
					acc_m_ema_std_rsf.append(np.mean(np.array(acc_m_ema_std)))

				if args.wandb:
					if len(acc_rsf) > 0:  # 为RSF汇总添加保护
						wandb.log({
	                        "cross_val/acc_mean": np.mean(np.array(acc_rsf)),
	                        "cross_val/auc_mean": np.mean(np.array(auc_rsf)),
	                        "cross_val/f1_mean": np.mean(np.array(fs_rsf)),
	                        "cross_val/acc_std_mean": np.mean(np.array(acc_std_rsf)),
	                        "cross_val/auc_std_mean": np.mean(np.array(auc_std_rsf)),
	                        "cross_val/f1_std_mean": np.mean(np.array(fs_std_rsf)),
	                        "cross_val/pre_mean": np.mean(np.array(pre_rsf)),
	                        "cross_val/recall_mean": np.mean(np.array(rec_rsf)),
	                        "cross_val/ck_mean": np.mean(np.array(ck_rsf)),
	                        "cross_val/acc_micro_mean": np.mean(np.array(acc_m_rsf)),
	                        "cross_val/acc_std": np.std(np.array(acc_rsf)),
	                        "cross_val/auc_std": np.std(np.array(auc_rsf)),
	                        "cross_val/f1_std": np.std(np.array(fs_rsf)),
	                    })
					if len(acc_ema_rsf) > 0:  # 为EMA RSF汇总添加保护
						wandb.log({
	                        "cross_val/acc_ema_mean": np.mean(np.array(acc_ema_rsf)),
	                        "cross_val/auc_ema_mean": np.mean(np.array(auc_ema_rsf)),
	                        "cross_val/f1_ema_mean": np.mean(np.array(fs_ema_rsf)),
	                        "cross_val/pre_ema_mean": np.mean(np.array(pre_ema_rsf)),
	                        "cross_val/recall_ema_mean": np.mean(np.array(rec_ema_rsf)),
	                        "cross_val/ck_ema_mean": np.mean(np.array(ck_ema_rsf)),
	                        "cross_val/acc_micro_ema_mean": np.mean(np.array(acc_m_ema_rsf)),
	                        "cross_val/acc_ema_std_mean": np.mean(np.array(acc_ema_std_rsf)),
	                        "cross_val/auc_ema_std_mean": np.mean(np.array(auc_ema_std_rsf)),
	                        "cross_val/f1_ema_std_mean": np.mean(np.array(fs_ema_std_rsf)),
	                        "cross_val/acc_m_ema_std_mean": np.mean(np.array(acc_m_ema_std_rsf)),
	                        "cross_val/acc_ema_std": np.std(np.array(acc_ema_rsf)),
	                        "cross_val/auc_ema_std": np.std(np.array(auc_ema_rsf)),
	                        "cross_val/f1_ema_std": np.std(np.array(fs_ema_rsf)),
	                    })
    
				if len(acc_rsf) > 0:  # 为最终打印添加保护
					print('Cross validation accuracy mean: %.4f, std %.4f ' %
	                          (np.mean(np.array(acc_rsf)), np.std(np.array(acc_rsf))))
					print('Cross validation auc mean: %.4f, std %.4f ' %
	                          (np.mean(np.array(auc_rsf)), np.std(np.array(auc_rsf))))
					print('Cross validation precision mean: %.4f, std %.4f ' %
	                          (np.mean(np.array(pre_rsf)), np.std(np.array(pre_rsf))))
					print('Cross validation recall mean: %.4f, std %.4f ' %
	                          (np.mean(np.array(rec_rsf)), np.std(np.array(rec_rsf))))
					print('Cross validation f1 score mean: %.4f, std %.4f ' %
	                          (np.mean(np.array(fs_rsf)), np.std(np.array(fs_rsf))))
				else:
					print('Warning: No RSF metrics collected')
            
		else:
			main(args=args,device=device)
	except:
		traceback.print_exc()
	finally:
		pass
