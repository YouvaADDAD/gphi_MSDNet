import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from typing import Union, List, Any
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def compute_compulative_probs(probs):
    B,_ = probs.shape

    probs = probs

    log_probs = torch.log(probs + 1e-5)
    log_neg_probs = torch.log1p(-probs + 1e-5)
    log_probs = torch.cat((log_probs,
                           torch.zeros((B,1), dtype=probs.dtype, layout=probs.layout, device=probs.device)), dim = -1)
        
    log_neg_probs = torch.cat((torch.zeros((B,1), dtype=probs.dtype, layout=probs.layout, device=probs.device),
                                log_neg_probs), dim = -1)

    log_neg_probs = log_neg_probs.cumsum(dim=-1)
    
    score_probs_prod =  torch.exp((log_probs + log_neg_probs)/0.5)
    # score_probs_prod = score_probs_prod/score_probs_prod.sum(dim=-1,keepdim=True)

    return score_probs_prod

def train_one_epoch(model: torch.nn.Module, meta_net: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, meta_optimizer: torch.optim.Optimizer, costs: torch.Tensor, alpha_cost,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, meta_lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, calibrator_interval=20):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if data_iter_step > 20:
            # break
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        if meta_lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(meta_optimizer.param_groups):
                param_group["lr"] = meta_lr_schedule_values[it] #* param_group["lr_scale"]



        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        if mixup_fn is not None:
            samples, softtargets = mixup_fn(samples, targets)

        ###############################New Optimization############################################
        if False:#epoch<calibrator_interval:
            model.train()
            output = model(samples)

            if not isinstance(output, list):
                output = [output]

            loss_all = [criterion(output[j], softtargets).mean() for j in range(len(output))]
           
            loss = sum(loss_all) 
            
            loss_value = loss.item()

            if not math.isfinite(loss_value): # this could trigger if using AMP
                print("Loss is {}, stopping training".format(loss_value))
                assert math.isfinite(loss_value)
            
            loss /= update_freq
            loss.backward()

            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        
        else:
            images_p1, images_p2 = samples.chunk(2, dim=0)
            softtargets_p1, softtargets_p2 = softtargets.chunk(2, dim=0)
            #target_p1, target_p2 = targets.chunk(2, dim=0)

            ###################################################
            ## part 1: images_p1 as train, images_p2 as meta ##
            ###################################################
            model.train()
            output = model(images_p1)


            #with torch.no_grad():
            probs_per_exit = []
            for j in range(len(output)-1): 
                previous_uncertainty = utils.compute_uncertainty(output[j])
                uncertainty = torch.stack(previous_uncertainty, dim=-1)
                probs_per_exit.append(meta_net.module[j](uncertainty))

            probs_per_exit = torch.cat(probs_per_exit, dim=-1)#(B,K)

            g_phi = compute_compulative_probs(probs_per_exit)
            g_phi = g_phi/g_phi.sum(dim=-1, keepdim=True)
            # g_phi = torch.exp(g_phi/0.8)
            # g_phi = g_phi/g_phi.sum(dim=-1, keepdim=True)


            loss_per_exit = torch.stack([criterion(output[l], softtargets_p1) for l in range(len(output))], dim=-1)
            loss_weighted = (loss_per_exit + alpha_cost * costs) * (g_phi + 1) 
            loss = loss_weighted.mean(dim=0).sum() #+ ( 0.5 * criterion(output[-1], softtargets_p2)).mean()


            loss_value = loss.item()

        
            if not math.isfinite(loss_value): # this could trigger if using AMP
                print("Loss is {}, stopping training".format(loss_value))
                assert math.isfinite(loss_value), print(g_phi)


            loss /= update_freq
            optimizer.zero_grad()
            loss.backward()

            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            
            if step%10==0:
                model.eval()
                with torch.no_grad():
                    output = model(images_p2)

                if not isinstance(output, list):
                        output = [output]

                probs_per_exit = []
                for j in range(len(output)-1): 
                    uncertainty = utils.compute_uncertainty(output[j])
                    uncertainty = torch.stack(uncertainty, dim=-1)
                    probs_per_exit.append(meta_net.module[j](uncertainty))

                probs_per_exit = torch.cat(probs_per_exit, dim=-1)#(B,K)

                g_phi = compute_compulative_probs(probs_per_exit)
                g_phi = g_phi/g_phi.sum(dim=-1, keepdim=True)
                # g_phi = torch.exp(g_phi/0.8)
                # g_phi = g_phi/g_phi.sum(dim=-1, keepdim=True)

 
                loss_per_exit_ce_and_cost = torch.stack([criterion(output[l], softtargets_p2) for l in range(len(output))], dim=-1)
                loss_per_exit_ce_and_cost = (loss_per_exit_ce_and_cost + alpha_cost * costs) * g_phi 
                loss = loss_per_exit_ce_and_cost.mean(dim=0).sum() #+ (0.5 * criterion(output[-1], softtargets_p2)).mean()


                    
                if not math.isfinite(loss_value): # this could trigger if using AMP
                    print("Loss is {}, stopping training".format(loss_value))
                    assert math.isfinite(loss_value), print(g_phi)


                loss /= update_freq
                #print(loss)
                meta_optimizer.zero_grad()
                loss.backward()


                if (data_iter_step + 1) % update_freq == 0:
                    meta_optimizer.step()
                    meta_optimizer.zero_grad()
                

                #exit()


            ###################################################
            ## part 1: images_p2 as train, images_p1 as meta ##
            ###################################################
            model.train()
            output = model(images_p2)


            #with torch.no_grad():
            probs_per_exit = []
            for j in range(len(output)-1): 
                previous_uncertainty = utils.compute_uncertainty(output[j])
                #previous_uncertainty = (previous_uncertainty, uncertainty) if uncertainty is not None else previous_uncertainty
                uncertainty = torch.stack(previous_uncertainty, dim=-1)
                probs_per_exit.append(meta_net.module[j](uncertainty))

            probs_per_exit = torch.cat(probs_per_exit, dim=-1)#(B,K)
            


            g_phi = compute_compulative_probs(probs_per_exit)
            g_phi = g_phi/g_phi.sum(dim=-1, keepdim=True)
            # g_phi = torch.exp(g_phi/0.8)
            # g_phi = g_phi/g_phi.sum(dim=-1, keepdim=True)
                
                
            loss_per_exit = torch.stack([criterion(output[l], softtargets_p2) for l in range(len(output))], dim=-1)
            loss_weighted = (loss_per_exit + alpha_cost * costs) * (g_phi + 1) #+  criterion(output[-1], target_p1).mean()#* g_phi #* len(output) #+ alpha_cost * costs


            loss = loss_weighted.mean(dim=0).sum() #+ (0.5 * criterion(output[-1], softtargets_p2)).mean()


            loss_value = loss.item()

        
            if not math.isfinite(loss_value): # this could trigger if using AMP
                print("Loss is {}, stopping training".format(loss_value))
                assert math.isfinite(loss_value), print(g_phi)


            loss /= update_freq
            ##Pas de prise de risque
            optimizer.zero_grad()
            #####################

            loss.backward()

            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            
            if step%10==0:
                model.eval()
                with torch.no_grad():
                    output = model(images_p1)

                if not isinstance(output, list):
                        output = [output]

                probs_per_exit = []
                for j in range(len(output)-1): 
                    uncertainty = utils.compute_uncertainty(output[j])
                    uncertainty = torch.stack(uncertainty, dim=-1)
                    probs_per_exit.append(meta_net.module[j](uncertainty))

                probs_per_exit = torch.cat(probs_per_exit, dim=-1)#(B,K)
                
                g_phi = compute_compulative_probs(probs_per_exit)
                g_phi = g_phi/g_phi.sum(dim=-1, keepdim=True)
                # g_phi = torch.exp(g_phi/0.8)
                # g_phi = g_phi/g_phi.sum(dim=-1, keepdim=True)

 
                loss_per_exit_ce_and_cost = torch.stack([criterion(output[l], softtargets_p1) for l in range(len(output))], dim=-1)
                loss_per_exit_ce_and_cost = (loss_per_exit_ce_and_cost + alpha_cost * costs) * g_phi
                loss = loss_per_exit_ce_and_cost.mean(dim=0).sum() #+ (0.5 * criterion(output[-1], softtargets_p2)).mean()
                    
                if not math.isfinite(loss_value): # this could trigger if using AMP
                    print("Loss is {}, stopping training".format(loss_value))
                    assert math.isfinite(loss_value), print(g_phi)


                loss /= update_freq
                meta_optimizer.zero_grad()
                loss.backward()


                if (data_iter_step + 1) % update_freq == 0:
                    meta_optimizer.step()
                    meta_optimizer.zero_grad()

        ###############################New Optimization############################################

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output[-1].max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        meta_min_lr = 10.
        meta_max_lr = 0.
        for group in meta_optimizer.param_groups:
            meta_min_lr = min(meta_min_lr, group["lr"])
            meta_max_lr = max(meta_max_lr, group["lr"])

        metric_logger.update(meta_lr=meta_max_lr)
        metric_logger.update(meta_min_lr=meta_min_lr)


        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    i = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        i += 1
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if type(output) is dict:
                    output = output['main']
                if not isinstance(output, list):
                    output = [output]
                loss = sum([criterion(output[j], target) for j in range(len(output))])
        else:
            output = model(images)
            if type(output) is dict:
                output = output['main']
            if not isinstance(output, list):
                output = [output]
            loss = sum([criterion(output[j], target) for j in range(len(output))])/len(output)

        lcl = dict()
        for j in range(len(output)):
            acc1, acc5 = accuracy(output[j], target, topk=(1, 5))
            lcl[f'acc1_clf{j}'] = acc1
            lcl[f'acc5_clf{j}'] = acc5


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        for j in range(len(output)):
            metric_logger.meters[f'acc1_clf{j}'].update(lcl[f'acc1_clf{j}'].item(), n=batch_size)
            metric_logger.meters[f'acc5_clf{j}'].update(lcl[f'acc5_clf{j}'].item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for j in range(len(output)):
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.meters[f'acc1_clf{j}'], top5=metric_logger.meters[f'acc5_clf{j}'], losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}