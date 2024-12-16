#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import datetime
import shutil

import argparse
from args import arg_parser
from adaptive_inference import dynamic_evaluate
from models.msdnet import MSDNet
import models
from op_counter import measure_model, get_flops_params

import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, create_meta_optimizer
from datasets import build_dataset
from engine import train_one_epoch, evaluate, compute_compulative_probs
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils

from models.weightModel import MLP_sigmoid

def main(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('number of device : ', torch.cuda.device_count())
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    print('The current device is: ', device)
    if args.input_size is None :
        if args.data_set.startswith('CIFAR'):
            args.input_size = 32
        else:
            args.input_size = 224
    if args.test_size <= 0:
        args.test_size = args.input_size
    
    print('input size : ', args.input_size, ' test size : ', args.test_size)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

    print(f'Number of Examples in Training Dataset is : {len(dataset_train)}')

    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)
        print(f'Number of Examples in Validation Dataset is : {len(dataset_val)}')

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    print('global rank ', global_rank)
    print(f'number of all tasks is : {num_tasks}')

    if utils.is_dist_avail_and_initialized():
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    
    print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    print("sampler_val = %s" % str(sampler_val))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    
    data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,)
    
    print(f'Training dataloader length is : {len(data_loader_train)}')

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size), #1.5 * 
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        print(f'Validation dataloader length is : {len(data_loader_val)}')
    else:
        data_loader_val = None
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    #   ================================================== build model ================================ #
    model = getattr(models, args.arch)(args)
    n_flops, n_params = get_flops_params(model, args.input_size)

    costs = torch.as_tensor(n_flops, device=device)
    costs = costs/costs[-1]

    torch.save(n_flops, os.path.join(args.log_dir, 'flops.pth'))
    del(model)

    print("-----------------------------------------")
    model = getattr(models, args.arch)(args)
    meta_net = nn.ModuleList([MLP_sigmoid(input_size=2, hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers, output_size=1) for _ in range(args.nBlocks - 1)])

    model.to(device)
    meta_net.to(device)

    model_ema = None

    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
        #model_ema.ema to have the model without ddp


    model_without_ddp = model
    meta_net_without_ddp = meta_net


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)
    print("-----------------------------------------")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        meta_net = torch.nn.parallel.DistributedDataParallel(meta_net, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
        meta_net_without_ddp = meta_net.module

    

    optimizer = create_optimizer(args, model_without_ddp, skip_list=None,)

    meta_optimizer = create_meta_optimizer(args, meta_net_without_ddp, skip_list=None, filter_bias_and_bn=False)
    #print(optimizer)

    loss_scaler = NativeScaler()  # if args.use_amp is False, this won't be used
    
    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )


    meta_lr_schedule_values = utils.cosine_scheduler(
        args.meta_lr, args.meta_min_lr, args.epochs, num_training_steps_per_epoch,  
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    print("-----------------------------------------")
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    print('=============================== GPU memory before AUTO LOAD', torch.cuda.memory_allocated())

    
    
    #####optimizer = create_optimizer(args, model_without_ddp, skip_list=None,)



    #print('=============================== GPU memory after AUTO LOAD', torch.cuda.memory_allocated())

    #utils.optimizer_to(optimizer, device)

    #print('=============================== GPU memory after moving optim to GPU', torch.cuda.memory_allocated())


    




    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = utils.SoftTargetCrossEntropy(reduction='none')
        #criterion = nn.HuberLoss()
    elif args.smoothing > 0.:
        criterion = utils.LabelSmoothingCrossEntropy(smoothing=args.smoothing, reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, meta_net=meta_net, model_without_ddp=model_without_ddp,meta_net_without_ddp=meta_net_without_ddp,
        optimizer=optimizer, meta_optimizer=meta_optimizer, loss_scaler=loss_scaler, model_ema=model_ema)


    if args.eval:
        if args.data_set.startswith('cifar'):
            num_samples = 5000
        else:
            num_samples = 50000
        print(f"Eval only mode")
        print('dynamic evaluate')
        #dynamic_evaluate(model, data_loader_val, data_loader_train, 'dynamic.txt', args)
        train_set_index = torch.randperm(len(dataset_train))
        data_loader_train = torch.utils.data.DataLoader(
                dataset_train, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                train_set_index[-num_samples:]),
                num_workers=args.num_workers, pin_memory=True)
        
        print(f"DataLoader Train Size : {len(data_loader_train)}")
        print(f"DataLoader Validation Size : {len(data_loader_val)}")
        
        dynamic_evaluate(model, data_loader_val, data_loader_train, args)
        # test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        # for j in range(args.nBlocks):
        #         print(f"Accuracy of the model on the {len(dataset_val)} test images and Classifier {j} is : {test_stats[f'acc1_clf{j}']:.1f}%")
        return


    max_accuracy = 0.0

    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0
    print('alpha for cost is ', args.alpha_cost)
    print("costs flops is", costs)
    
    print('=============================== GPU memory before training', torch.cuda.memory_allocated())

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        
        train_stats = train_one_epoch(
            model, meta_net, criterion, data_loader_train, optimizer, meta_optimizer, costs, args.alpha_cost,
            device, epoch, loss_scaler, max_norm = args.clip_grad, model_ema = model_ema, mixup_fn = mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, meta_lr_schedule_values=meta_lr_schedule_values,wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )

        if args.save and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, meta_net=meta_net, model_without_ddp=model_without_ddp, meta_net_without_ddp=meta_net_without_ddp,optimizer=optimizer,
                    meta_optimizer=meta_optimizer,loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
            utils.save_model(
                args=args, model=model, meta_net=meta_net, model_without_ddp=model_without_ddp, meta_net_without_ddp=meta_net_without_ddp, optimizer=optimizer,
                meta_optimizer=meta_optimizer, loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, is_latest=True)
        
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            for j in range(args.nBlocks):
                print(f"Accuracy of the model on the {len(dataset_val)} test images and Classifier {j} is : {test_stats[f'acc1_clf{j}']:.1f}%")

            if max_accuracy < test_stats[f"acc1_clf{args.nBlocks-1}"]:
                max_accuracy = test_stats[f"acc1_clf{args.nBlocks-1}"]
                if args.save and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, meta_net=meta_net, model_without_ddp=model_without_ddp, meta_net_without_ddp=meta_net_without_ddp, optimizer=optimizer,
                        meta_optimizer=meta_optimizer, loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')
            
            if log_writer is not None:
                for j in range(args.nBlocks):
                    log_writer.update(test_acc1=test_stats[f"acc1_clf{j}"], head="perf", step=epoch)
                    log_writer.update(test_acc5=test_stats[f"acc5_clf{j}"], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                for j in range(args.nBlocks):
                    print(f"Accuracy of the model EMA on the {len(dataset_val)} test images and Classifier {j} is : {test_stats_ema[f'acc1_clf{j}']:.1f}%")
                
                if max_accuracy_ema < test_stats_ema[f"acc1_clf{args.nBlocks-1}"]:
                    max_accuracy_ema = test_stats_ema[f"acc1_clf{args.nBlocks-1}"]
                    if args.save and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, meta_net=meta_net, model_without_ddp=model_without_ddp, meta_net_without_ddp=meta_net_without_ddp, optimizer=optimizer,
                            meta_optimizer=meta_optimizer, loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})

        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        
        if args.save and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.save, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    











if __name__ == '__main__':
    args = arg_parser.parse_args()

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.nScales = len(args.grFactor)
    args.splits = ['train', 'val'] #Use only train and val
    args.data_set = args.data_set.lower()

    #Create only one file not #nb_process file
    if args.save and utils.is_main_process():
        Path(args.save).mkdir(parents=True, exist_ok=True)
    
    if args.log_dir and utils.is_main_process():
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)