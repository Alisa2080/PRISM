import argparse
import yaml
import torch

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Computional Pathology Training Script')    

group = parser.add_argument_group('Dataset')
group.add_argument('--data_root_dir',   type=str, default=r'/media/miku/code/code/PORPOISE-master', help='Data directory to WSI features (extracted via CLAM)')
group.add_argument('--num_splits', 	 type=int, default=5, help='Number of splits (default: 5)')
group.add_argument('--cancer_type',     type=str, default='GBM_LGG', help='Cancer type')
group.add_argument("--gene_dir",        type=str, default=r"/media/miku/code/code/PORPOISE-master/csv/classification/gbmlgg_signatures.csv", help="Data directory to signature")
group.add_argument("--evaluate",        action="store_true", dest="evaluate", help="Evaluate model on test set")
group.add_argument("--num_pathway",     type=int, default=282, help="Maximum number of pathways")
group.add_argument("--weighted_sample", action="store_true", default=True, help="Enable weighted sampling")
group.add_argument("--task",            type=str, default="classification", choices=["survival", "classification", "grading"], help="Task type: survival, classification, or grading")
group.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
group.add_argument('--random_seed_fold', default=0, type=int, help='Random seed used when creating cross-validation folds')
group.add_argument('--random_fold', action='store_true', help='Muti-fold random experiment')
# Patch size settings
group.add_argument('--same_psize', default=0., type=float, help='Keep the same size of all patches [0]')
group.add_argument('--same_psize_pad_type', default='zero', type=str, choices=['zero', 'random', 'none'])
group.add_argument('--same_psize_ratio', default=0., type=float, help='Keep the same ratio of all patches [0]')
# Dataloader settings
group.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
group.add_argument('--pin_memory', action='store_true', help='Enable Pinned Memory')
group.add_argument('--file_worker', action='store_true', help='Enable file system sharing via workers')
group.add_argument('--no_prefetch', action='store_true', help='Disable prefetching')
group.add_argument('--prefetch_factor', default=2, type=int, help='Prefetch factor [2]')
# Model Parameters.
group = parser.add_argument_group('Training')

group.add_argument('--epoch_start', default=0, type=int, help='Epoch index to resume training from')
group.add_argument('--num_epoch', default=75, type=int, help='Number of total training epochs [200]')
group.add_argument('--early_stopping', action='store_false', help='Early stopping')
group.add_argument('--max_epoch', default=30, type=int, help='Number of max training epochs in the earlystopping [130]')
group.add_argument('--warmup_epochs', default=5, type=int, help='Number of training epochs with warmup lr')
group.add_argument('--patient', default=20, type=int, help='Patience (in epochs) before early stopping is triggered')
group.add_argument('--batch_size', default=2, type=int, help='Number of batch size')
group.add_argument('--loss', default='ce', type=str, choices=['ce', 'bce', 'asl', 'nll_surv'], help='Classification Loss, defualt nll_surv in survival prediction [ce, bce, nll_surv]')
group.add_argument('--label_smooth', default=0., type=float, help='Label smoothing factor')
group.add_argument('--opt', default='adamw', type=str, help='Optimizer [adam, adamw]')
group.add_argument('--model', default='cmta', type=str, help='Model name')
group.add_argument('--seed', default=2021, type=int, help='random number [2021]')
group.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
group.add_argument('--lr_base_size', default=1, type=int, help='Reference batch size used for learning-rate scaling')
group.add_argument('--warmup_lr', default=1e-6, type=float, help='Starting learning rate during the warmup phase')
group.add_argument('--lr_sche', default='cosine', type=str, help='Decay of learning rate [cosine, step, const]')
group.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
group.add_argument('--lr_scale', action='store_true', help='Scale learning rate according to global-to-base batch size ratio')
group.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
group.add_argument('--num_classes',     type=int, default=4) ###############
group.add_argument('--use_class_weights', action='store_true', help='Use class weights from splits for loss re-weighting')
group.add_argument('--class_weights_from', type=str, default='fold', choices=['fold','overall'], help='Use per-fold or overall weights when available')

group.add_argument('--main_alpha', default=1.0, type=float, help='Main loss alpha')
group.add_argument('--aux_alpha', default=0.8, type=float, help='Auxiliary loss alpha')
group.add_argument('--batch_loss_alpha', default=1.0, type=float, help='Batch structural loss alpha')

# MultimodalBatchLoss hyperparameters
group.add_argument('--use_batch_loss', action='store_true', default=False, help='Enable BatchLoss for structural alignment')
group.add_argument('--batch_loss_mem_size', default=512, type=int, help='Cross-batch memory size for structural loss')
group.add_argument('--batch_loss_shrink', default=0.1, type=float, help='Shrinkage lambda (0.05–0.1 recommended)')
group.add_argument('--batch_loss_mem_mem_weight', default=0.0, type=float, help='Weight for memory×memory block in structural loss')

# EMA teacher targets for structural alignment
group.add_argument('--batch_loss_use_ema', action='store_true', help='Use EMA teacher targets for structural alignment')
group.add_argument('--batch_loss_ema_decay', default=0.99, type=float, help='EMA decay for teacher encoders')
group.add_argument('--batch_loss_ema_weight', default=0.8, type=float, help='Weight for teacher-target alignment term')

group.add_argument('--test_type', default='main', type=str, choices=['main', 'ema', 'both', 'both_ema'])
group.add_argument('--model_ema', action='store_true', help='Enable Model EMA')
group.add_argument('--mm', type=float, default=0.9999, help='Model EMA decay rate (often called momentum)')
group.add_argument('--mm_sche', action='store_true', default=False, help='Enable warmup for model EMA decay')
group.add_argument('--bin_metric', action='store_true', help='Use binary average when n_classes==2')
group.add_argument('--no_determ', action='store_true', help='Disable PyTorch deterministic mode (enables CuDNN benchmark)')
group.add_argument('--no_deter_algo', action='store_true', help='Allow non-deterministic algorithms when determinism is enabled')
group.add_argument('--no_drop_last', action='store_true', default=False, help='Keep the last incomplete batch instead of dropping it')
group.add_argument('--empty_cuda_cache', action='store_true', default=False, help='Clear CUDA cache')
group.add_argument('--accumulation_steps', default=3, type=int, help='Gradient accumulation steps')
group.add_argument('--always_test', action='store_true', help='Test model in the training phase')

group = parser.add_argument_group('Evaluate')
group.add_argument('--all_test', action='store_true', help='Evaluate using the union of train, validation, and test splits')
group.add_argument('--num_bootstrap', default=1000, type=int, help='Number of bootstrap samples for metric estimation')
group.add_argument('--bootstrap_mode', default='test', type=str, choices=['test', 'none', 'val', 'test_val'])# Model

group = parser.add_argument_group('Model')
group.add_argument('--pack_bs', action='store_true', help='use packmil')
group.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')
group.add_argument('--inner_dim', default=256, type=int, help='Hidden dimension for the MIL backbone')
group.add_argument('--act', default='gelu', type=str, choices=['relu', 'gelu', 'none'],
                   help='Activation func in the projection head [gelu,relu]')
group.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
group.add_argument('--cls_norm', default='ln', choices=['bn', 'ln', 'none'])
group.add_argument('--mil_norm', default='ln', choices=['bn', 'ln', 'none'])
group.add_argument('--no_mil_bias', action='store_true', help='DSMIL hyperparameter')
group.add_argument('--pos', default='ppeg', type=str, choices=['ppeg', 'sincos', 'alibi', 'none', 'alibi_learn', 'rope'])
group.add_argument('--embed_norm_pos', default=0, type=int, help='Position of normalization in the feature embed (0=input, 1=hidden)')
group.add_argument('--mil_feat_embed_mlp_ratio', default=4, type=int, help='Expansion ratio for the MIL feature embedding MLP')

group.add_argument('--max_patches', default=8000, type=int, help='Maximum number of patches/ROIs per slide')

group.add_argument('--patch_shuffle', action='store_true', help='Patch shuffle')
group.add_argument('--n_layers', default=2, type=int, help='Number of transformer layers in the MIL backbone')
group.add_argument('--num_heads', default=4, type=int, help='Number of head in the MSA')
group.add_argument('--pool', default=None, type=str, help='Pooling strategy applied to MIL encoder outputs')
group.add_argument('--attn_dropout', default=0., type=float, help='Dropout rate applied inside attention blocks')
group.add_argument('--attn_type', default='sa', type=str, choices=['sa', 'ca', 'ntrans'], help='Type of attention mechanism to use in transformer blocks. [Only for Ablation]')
group.add_argument('--sdpa_type', default='torch', type=str, choices=['torch', 'flash', 'math', 'memo_effi', 'torch_math', 'ntrans'])

group.add_argument('--da_gated', action='store_true', help='DSMIL hyperparameter')

group.add_argument('--token_dropout', default=0.5, type=float, help='Drop ratio applied when sampling tokens for packing')
group.add_argument('--pack_residual_downsample_r', default=4, type=int, help='Downsample ratio used by the residual branch')
group.add_argument('--pack_residual_type', default="norm", type=str, choices=['norm', 'dual_cls'], help='whether to use a separate classifier for the residual branch. default not to use a separate classifier.')
group.add_argument('--pack_pad_r', action='store_true', help='only for survival task, pad the sequence to be divisible by pack_residual_downsample_r')
group.add_argument('--min_seq_len', default=256, type=int, help='Minimum sequence length enforced after packing')
group.add_argument('--pack_max_seq_len', default=3200, type=int, help='Maximum packed sequence length per bag')
group.add_argument('--pack_residual_loss', default="focal", type=str, choices=['bce', 'ce', 'nll', 'asl_single', 'focal'])
group.add_argument('--pack_downsample_mode', default="ads", type=str, choices=['none','ads'], help='Strategy used to downsample sequences inside PackMIL')
group.add_argument('--pack_downsample_type', default="random", type=str, choices=['random','max'], help='Random for subtype, max for survival. max is more easy to converge. Pooling') 
group.add_argument('--pack_residual_ps_weight', action='store_true', help='Weight residual targets by the number of kept patches')
group.add_argument('--pack_singlelabel', action='store_true', help='Treat residual supervision as single-label targets')
group.add_argument('--pack_no_residual', action='store_true', help='[Only for Ablation] Disable residual branch to learn from dropped tokens')
group.add_argument('--fusion',          type=str, choices=['concat', 'bilinear'], default='concat', help='Type of fusion. (Default: bilinear).')

group = parser.add_argument_group('Miscellaneous')
group.add_argument('--title', default='pa-GBM-CLS-itc', type=str, help='Title of exp')
group.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
group.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
group.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--amp_test', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--amp_scale_index', default=16, type=int, help='Automatic Mixed Precision Training')
group.add_argument('--amp_growth_interval', default=2000, type=int, help='Automatic Mixed Precision Training')
group.add_argument('--amp_unscale', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--output_path', type=str, default=r"/media/miku/code/code/PORPOISE-master/results",help='Output path')
group.add_argument('--model_path', default=None, type=str, help='model init path')
group.add_argument('--script_mode', default='all', type=str, help='[all, no_train, test, only_train]')
group.add_argument('--profile', action='store_true', help='Enable torch.profiler tracing during training iterations')
group.add_argument('--torchcompile', action='store_true', help='Torch Compile for torch > 2.0')
group.add_argument('--torchcompile_mode', default='default', type=str,
                   choices=['default', 'reduce-overhead', 'max-autotune'], help='Compilation mode to pass into torch.compile')

group.add_argument('--wandb', action='store_true', help='Log metrics to Weights & Biases')
group.add_argument('--wandb_watch', action='store_true', help='Watch model parameters/gradients in Weights & Biases')
group.add_argument('--save_iter', default=-1, type=int, help='Checkpoint saving interval in epochs (-1 to disable)')

def _parse_args():
    # 1. 第一阶段：预解析，只寻找 --config 参数
    args_config, remaining = config_parser.parse_known_args()

    cfg = {}
    # 2. 第二阶段：加载 YAML 文件（如果提供了）
    if args_config.config:
        config_files = args_config.config.split(',')
        for config_file in config_files:
            config_file = config_file.strip()  
            if config_file: 
                try:
                    with open(config_file, 'r') as f:
                        cfg.update(yaml.safe_load(f))
                except Exception as e:
                    print(f"Error loading config file {config_file}: {str(e)}")

        # 3. 第三阶段：将 YAML 配置设置为新的默认值
        parser.set_defaults(**cfg)

    # 4. 第四阶段：解析剩余的命令行参数
    args = parser.parse_args(remaining)
    args.config = args_config.config.split(',')

    # 5. 第五阶段：生成配置的文本表示
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text


def more_about_config(args):
    # more about config

    args.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.device = device = init_distributed_device(args)
    # train
    # args.mil_bias = not args.no_mil_bias
    args.drop_last = not args.no_drop_last
    args.prefetch = not args.no_prefetch
    args.mil_feat_embed = True
    args.mil_bias = not args.no_mil_bias
    args.mil_feat_embed_type = "norm"
    args.seed_ori = args.seed
    if not args.amp_test:
        args.amp_test = args.amp

    # if args.pos in ('sincos', 'alibi', 'alibi_learn', 'rope'):
    #     assert args.h5_path is not None

    # if args.persistence:
    #     if args.same_psize > 0:
    #         raise NotImplementedError("Random same patch is different from not presistence")

    # if args.val2test or args.val_ratio == 0.:
    #     args.always_test = False

    # if args.pos in ('sincos', 'alibi') and args.h5_path is None:
    #     raise NotImplementedError

    # if args.mil_feat_embed:
    #     args.mil_feat_embed = ~(args.mil_feat_embed_type == 'none')

    # if args.datasets.lower() == 'panda':
    #     args.n_classes = 6

    if args.model == 'mhim_pure':
        args.aux_alpha = 0.
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model in ('clam_sb', "clam_mb"):
        args.main_alpha = .7
        args.aux_alpha = .3
    elif args.model == 'dsmil':
        if args.main_alpha > 0.:
            args.main_alpha = 0.5
            args.aux_alpha_1 = 0.5
        else:
            args.aux_alpha_1 = 0.
        args.aux_alpha_2 = args.aux_alpha
        args.aux_alpha = 1

    # if args.model == '2dmamba':
    #     if args.datasets.lower().endswith('brca'):
    #         args.mamba_2d_max_w = 413
    #         args.mamba_2d_max_h = 821
    #     elif args.datasets.lower().endswith('panda'):
    #         args.mamba_2d_max_w = 384
    #         args.mamba_2d_max_h = 216
    #     elif args.datasets.lower().endswith('nsclc') or args.datasets.lower().endswith('luad') or args.datasets.lower().endswith('lusc'):
    #         args.mamba_2d_max_w = 385
    #         args.mamba_2d_max_h = 216
    #     elif args.datasets.lower().endswith('call'):
    #         args.mamba_2d_max_w = 432
    #         args.mamba_2d_max_h = 432
    #     elif args.datasets.lower().endswith('blca'):
    #         args.mamba_2d_max_w = 381
    #         args.mamba_2d_max_h = 275
    #     else:
    #         raise NotImplementedError(args.datasets)

    # multi-class cls, refer to top-1 acc, bin-class cls, refer to auc
    args.best_metric_index = 1 if args.num_classes != 2 and not args.task == 'survival' else 0

    args.max_epoch = min(args.max_epoch, args.num_epoch)

    return args, device
