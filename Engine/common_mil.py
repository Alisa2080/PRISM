
class CommonMIL():
	def __init__(self,args) -> None:
		self.training = True

	def init_func_train(self,args,**kwargs):
		self.training = True
        # Optionally reset cross-batch memories at epoch start
		model = kwargs.get('model', None)
		if getattr(args, 'use_batch_loss', False) and model is not None:
			try:
				target = model.module if hasattr(model, 'module') else model
				if hasattr(target, 'batch_loss_fn') and hasattr(target.batch_loss_fn, 'reset'):
					target.batch_loss_fn.reset()
			except Exception:
				pass
	
	def init_func_val(self,args,**kwargs):
		self.training = False
	
	def after_get_data_func(self,args,**kwargs):
		pass

	def forward_func(self,args,model,model_ema,bag,omic_data,label,criterion,batch_size,i,epoch,n_iter,pos,**kwargs):
		pad_ratio=0.
		kn_std = 0.
		if args.model == 'pack_pure':
			if args.baseline == 'dsmil':
				logits = model(bag)
				logits = 0.5*logits[0].view(batch_size,-1)+0.5*logits[1].view(batch_size,-1)
				aux_loss, patch_num, keep_num = 0., bag.size(1), bag.size(1)
			else:
				logits = model(bag, pos=pos)
				aux_loss, patch_num, keep_num = 0., bag.size(1), bag.size(1)
		elif args.model in ('clam_sb','clam_mb','dsmil'):
			
			if args.pack_bs:
				logits = model(bag,label=label,loss=criterion,pos=pos,epoch=epoch)
				(logits,aux_loss,_), _aux_loss, patch_num, keep_num, pad_ratio,kn_std = logits
				aux_loss = aux_loss*args.aux_alpha_1+_aux_loss*args.aux_alpha_2
			else:
				logits, aux_loss, _ = model(bag,label=label,loss=criterion,pos=pos)
				keep_num = patch_num = bag.size(1)
		else:
			if args.pack_bs:
				logits = model(bag,omic_data=omic_data,pos=pos,label=label,epoch=epoch)
				logits, aux_loss,batch_loss, patch_num, keep_num, pad_ratio,kn_std = logits
			else:
				logits = model(bag,omic_data=omic_data,pos=pos)
				aux_loss,batch_loss, patch_num, keep_num = 0., 0., bag.size(1), bag.size(1)

		return logits,label,aux_loss,batch_loss,patch_num,keep_num,pad_ratio,kn_std
	
	def after_backward_func(self,args,**kwargs):
        # Update internal EMA teachers if model provides the method
		model = kwargs.get('model', None)
		if model is not None:
			try:
				target = model.module if hasattr(model, 'module') else model
				if hasattr(target, 'update_ema_teachers'):
					target.update_ema_teachers()
			except Exception:
				pass
	
	def final_train_func(self,args,**kwargs):
		pass

	def validate_func(self,args,model,bag,omic_data,label,criterion,batch_size,i,pos,epoch=None,**kwargs):
		if args.model == 'dsmil':
			logits,_ = model(bag,pos=pos,epoch=epoch)
		elif args.model == 'pack_pure' and args.baseline == 'dsmil':
			logits,_ = model(bag,pos=pos,epoch=epoch)
		else:
			logits = model(bag,omic_data=omic_data,pos=pos,epoch=epoch)
		
		return logits,label
