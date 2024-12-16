import os
import glob
import time
import argparse

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
model_names = ['msdnet']

arg_parser = argparse.ArgumentParser(description='Training script for image classification')

#General training
arg_parser.add_argument('--epochs', default=300, type=int, metavar='N',
                         help='number of total epochs to run (default: 300)')

arg_parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

arg_parser.add_argument('-b', '--batch_size', default=64, type=int,
                         metavar='N', help='Per GPU batch size (default 64)')

arg_parser.add_argument('--use_stem',  default=True, type=str2bool,
                        help='Use Large Stem')  
                        
arg_parser.add_argument('--stem_channels',  default=64, type=int,
                        help='First Layer Number Of Channels (default 64)')  

arg_parser.add_argument('--repeat',  default=2, type=int,
                        help='Repeat Block repeat times (default 2)')


# Model parameters
arg_parser.add_argument('-a','--arch', default='fusionnet', metavar='MODEL', type=str, choices=model_names,
                        help='Name of model to train')
arg_parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
arg_parser.add_argument('--input_size', default=None, type=int,
                        help='image input size')
arg_parser.add_argument('--test_size', default=-1, type=int,
                        help='test input size')
arg_parser.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
# msdnet config
arg_parser.add_argument('--nBlocks', type=int, default=1)
arg_parser.add_argument('--nChannels', type=int, default=128)
arg_parser.add_argument('--OutChannels', default=128, type=int) 
arg_parser.add_argument('--base', type=int,default=4)
arg_parser.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
arg_parser.add_argument('--step', type=int, default=1)
arg_parser.add_argument('--growthRate', type=int, default=6)
arg_parser.add_argument('--grFactor', default='1-2-4', type=str)
arg_parser.add_argument('--prune', default='max', choices=['min', 'max'])
arg_parser.add_argument('--bnFactor', default='1-2-4')
arg_parser.add_argument('--bottleneck', default=True, type=bool) 


# Optimization parameters
arg_parser.add_argument('--opt', '--optimizer', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')

arg_parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')

arg_parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')

arg_parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

arg_parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

arg_parser.add_argument('--weight_decay','--wd', type=float, default=1e-4,
                        help='weight decay (default: 0.05)')

arg_parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")


arg_parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')

arg_parser.add_argument('--layer_decay', type=float, default=1.0)

arg_parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

arg_parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

arg_parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

arg_parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')


# Augmentation parameters
arg_parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
arg_parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
arg_parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
arg_parser.add_argument('--train_interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# Evaluation parameters
arg_parser.add_argument('--crop_pct', type=float, default=None)

# * Random Erase params
arg_parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
arg_parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
arg_parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
arg_parser.add_argument('--resplit', type=str2bool, default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
arg_parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0.')
arg_parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0.')
arg_parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
arg_parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
arg_parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
arg_parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')



# Dataset parameters
arg_parser.add_argument('--data_path', default='./data', type=str,
                help='dataset path')

arg_parser.add_argument('--eval_data_path', default=None, type=str,
                help='dataset path for evaluation')

arg_parser.add_argument('--nb_classes', default=1000, type=int,
                help='number of the classification types')

arg_parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)

arg_parser.add_argument('--data_set', default='cifar100', choices=['cifar10', 'cifar100', 'imagenet', 'image_folder'],
                type=str, help='Dataset Name')
arg_parser.add_argument('--save', default='save/',
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')

arg_parser.add_argument('--log_dir', default='log/',
                help='path where to tensorboard log')

arg_parser.add_argument('--device', default='cuda',
                help='device to use for training / testing')
arg_parser.add_argument('--seed', default=0, type=int)

arg_parser.add_argument('--resume', default='',
                help='resume from checkpoint')
arg_parser.add_argument('--auto_resume', type=str2bool, default=True)
arg_parser.add_argument('--save_ckpt', type=str2bool, default=True)
arg_parser.add_argument('--save_ckpt_freq', default=20, type=int)
arg_parser.add_argument('--save_ckpt_num', default=3, type=int)


arg_parser.add_argument('--eval', type=str2bool, default=False,
                help='Perform evaluation only')

arg_parser.add_argument('--split_val', type=str2bool, default=True,
                help='Split to train and validation set')

arg_parser.add_argument('--evalmode', default=None,
                       choices=['anytime', 'dynamic'],
                       help='which mode to evaluate')

arg_parser.add_argument('--dist_eval', type=str2bool, default=True,
                help='Enabling distributed evaluation')
arg_parser.add_argument('--disable_eval', type=str2bool, default=False,
                help='Disabling evaluation during training')
arg_parser.add_argument('--num_workers', default=10, type=int) #Change It in utils

arg_parser.add_argument('--pin_mem', type=str2bool, default=True,
                help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

# distributed training parameters
arg_parser.add_argument('--world_size', default=1, type=int,
                help='number of distributed processes')
arg_parser.add_argument('--local_rank', default=-1, type=int)
arg_parser.add_argument('--dist_on_itp', type=str2bool, default=False)
arg_parser.add_argument('--dist_url', default='env://',
                help='url used to set up distributed training')

arg_parser.add_argument('--use_amp', type=str2bool, default=False, 
                help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
arg_parser.add_argument('--use_checkpoint', type=str2bool, default=False,
                help="Use PyTorch's torch.util.checkpoint to save memory or not")

arg_parser.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')


# EMA related parameters
arg_parser.add_argument('--model_ema', type=str2bool, default=False)
arg_parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
arg_parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
arg_parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')



arg_parser.add_argument('--meta_net_hidden_size', default=100, type=int)
arg_parser.add_argument('--meta_net_num_layers', default=1, type=int)
arg_parser.add_argument('--meta_interval', default=1, type=int)
arg_parser.add_argument('--meta_lr', default=1e-3, type=float)
arg_parser.add_argument('--meta_min_lr', default=1e-6, type=float)
arg_parser.add_argument('--meta_weight_decay', type=float, default=1e-4)
arg_parser.add_argument('--alpha_cost', type=float, default=1e-1)








