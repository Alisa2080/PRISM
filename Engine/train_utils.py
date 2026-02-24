import torch
import torch.nn as nn
from timm.scheduler import create_scheduler_v2
import math
from Engine.utils import EarlyStopping
from timm.loss import AsymmetricLossSingleLabel

def adjust_encoder_learning_rate(model,n_epoch_warmup, n_epoch, max_lr, optimizer, dloader_len, step):
    """
    Set learning rate according to cosine schedule
    """
    max_steps = int(n_epoch * dloader_len)
    warmup_steps = int(n_epoch_warmup * dloader_len)
    step += 1

    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
        #lr = 0.
        # model.freeze_encoder()
    else:
        # model.unfreeze_encoder()
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = max_lr * 0.001
        lr = max_lr * q + end_lr * (1 - q)
        
    #print(optimizer.param_groups[0])
    optimizer.param_groups[0]['lr'] = lr

def zero_learning_rate(optimizer,mode='enc'):
    """
    Set learning rate according to cosine schedule
    """
        
    #print(optimizer.param_groups[0])
    if mode == 'enc':
        optimizer.param_groups[0]['lr'] = 0.
    elif mode == 'mil':
        #print(optimizer.param_groups[1])
        optimizer.param_groups[-1]['lr'] = 0.


############# Survival Prediction ###################
def nll_loss(hazards,S,Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1)  # censorship status, 0 or 1
    # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    
    return loss

from typing import Optional

class NLLSurvLoss(object):
    def __init__(self, alpha=0., class_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        self.alpha = alpha
        # class_weights is expected to be a 1D tensor of length = n_bins
        self.class_weights = class_weights
        self.reduction = reduction

    def __call__(self, Y, c, logits=None, hazards=None, S=None, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if hazards is None:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
        # Compute unweighted per-sample loss first
        batch_size = len(Y)
        Yv = Y.view(batch_size, 1)
        cv = c.view(batch_size, 1)
        S_padded = torch.cat([torch.ones_like(cv), S], 1)
        eps = 1e-7
        uncensored_loss = -(1 - cv) * (
            torch.log(torch.gather(S_padded, 1, Yv).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Yv).clamp(min=eps))
        )
        censored_loss = -cv * torch.log(torch.gather(S_padded, 1, Yv + 1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss_vec = (1 - alpha) * neg_l + alpha * uncensored_loss  # shape [B,1]

        # Apply class weights if provided (per-bin weights by Y)
        if self.class_weights is not None:
            # ensure tensor device/dtype matches loss tensor
            w = self.class_weights.to(loss_vec.device, loss_vec.dtype)
            sample_w = torch.gather(w.view(1, -1).expand(batch_size, -1), 1, Yv)
            loss_vec = loss_vec * sample_w

        if self.reduction == 'sum':
            return loss_vec.sum()
        else:
            return loss_vec.mean()


def build_train(args,model,**kwargs):
    # build criterion
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss(label_smoothing=args.label_smooth)
    elif args.loss == 'ce':
        # Optional class weights for imbalance mitigation
        if getattr(args, 'class_weights', None) is not None:
            cw = torch.as_tensor(args.class_weights, dtype=torch.float)
            # align to model device for safe CUDA loss computation
            try:
                dev = next(model.parameters()).device
                cw = cw.to(dev)
            except Exception:
                pass
            criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smooth)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    elif args.loss == "nll_surv":
        # Provide per-bin weights if available
        cw = None
        if getattr(args, 'class_weights', None) is not None:
            cw = torch.as_tensor(args.class_weights, dtype=torch.float)
        criterion = NLLSurvLoss(alpha=0.0, class_weights=cw)
    elif args.loss == 'asl':
        criterion = AsymmetricLossSingleLabel(gamma_neg=4, gamma_pos=1, eps=args.label_smooth)

    # lr scale
    if args.lr_scale:
        global_batch_size = args.batch_size * args.world_size * args.accumulation_steps
        batch_ratio = global_batch_size / args.lr_base_size
        batch_ratio = batch_ratio ** 0.5
        #batch_ratio = max(batch_ratio,4)
        lr = args.lr * batch_ratio
    else:
        lr = args.lr

    params = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr,'weight_decay': args.weight_decay},]
            
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params)
    elif args.opt == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(params)

    # build scheduler
    if args.lr_sche == 'cosine':
        scheduler,_ = create_scheduler_v2(optimizer,sched='cosine',num_epochs=args.num_epoch,warmup_lr=args.warmup_lr,warmup_epochs=args.warmup_epochs,min_lr=1e-7)

    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    # build early stopping
    if args.early_stopping:
        patience,stop_epoch = args.patient,args.max_epoch
        early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch)
    else:
        early_stopping = None
    
    return criterion,optimizer,scheduler,early_stopping
