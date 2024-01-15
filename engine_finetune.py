# --------------------------------------------------------
# References:
# SatMAE: https://github.com/sustainlab-group/SatMAE
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
import util.lr_sched as lr_sched
import util.misc as misc
import wandb
from timm.data import Mixup
from timm.utils import accuracy
import numpy as np
from sklearn.metrics import f1_score, jaccard_score


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
    ignore_index=-9999,
):
    if args is None:
        raise Exception("args is None")
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    accum_iter = args.accum_iter  # type: ignore
    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print(f"train_one_epoch: {samples.shape}, {targets.shape}")
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # print(f"train_one_epoch after mixup: {samples.shape}, {targets.shape}")
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb_project is not None:
                try:
                    wandb.log(
                        {
                            "train_loss_step": loss_value_reduce,
                            "train_lr_step": max_lr,
                            "epoch_1000x": epoch_1000x,
                        }
                    )
                except ValueError:
                    pass
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, ignore_index=-9999):
    criterion = torch.nn.CrossEntropyLoss()
    if args is None:
        raise Exception("args is None")
    # criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    true_labels = []
    predict = []
    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_one_hot = torch.zeros(
            (target.size(0), model.num_classes), device=target.device
        )
        target_one_hot.scatter_(1, target.unsqueeze(1).long(), 1)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        if int(args.nb_classes) < 4:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1 = accuracy(output, target, topk=(1,))
            if isinstance(acc1, list):
                acc1 = acc1[0]
            acc5 = None

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        if not args.use_psa:
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            if acc5 is not None:
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        true_labels.append(target_one_hot.cpu().numpy())  # store true labels
        predict.append(torch.argmax(output, dim=-1).cpu().numpy())

    y = np.argmax(np.concatenate(true_labels), axis=1).astype(int)
    predict = np.concatenate(predict).astype(int)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if not args.use_psa:
        macro_f1_score = f1_score(y, predict, average="macro")
        micro_f1_score = f1_score(y, predict, average="micro")
        metric_logger.add_meter("macro_f1", misc.AverageMeter())
        metric_logger.add_meter("micro_f1", misc.AverageMeter())
        metric_logger.update(macro_f1=macro_f1_score, micro_f1=micro_f1_score)

        # Log metrics to wandb
        if args.local_rank == 0 and args.wandb_project is not None:
            try:
                if int(args.nb_classes) < 4:
                    wandb.log(
                        {
                            "val_acc1": metric_logger.acc1.global_avg,
                            "val_acc5": metric_logger.acc5.global_avg,
                            "val_macro_f1": macro_f1_score,
                            "val_micro_f1": micro_f1_score,
                        }
                    )
                else:
                    wandb.log(
                        {
                            "val_acc1": metric_logger.acc1.global_avg,
                            "val_macro_f1": macro_f1_score,
                            "val_micro_f1": micro_f1_score,
                        }
                    )
            except ValueError:
                pass

        classwise_f1_score = f1_score(y, predict, average=None)
        if int(args.nb_classes) < 4:
            print(
                "* Acc@1 {top1.global_avg:.3f}\n* Acc@5 {top5.global_avg:.3f}\n*"
                " CE-loss {losses.global_avg:.3f}".format(
                    top1=metric_logger.acc1,
                    top5=metric_logger.acc5,
                    losses=metric_logger.loss,
                )
            )
        else:
            print(
                "* Acc@1 {top1.global_avg:.3f}\n* CE-loss {losses.global_avg:.3f}"
                .format(top1=metric_logger.acc1, losses=metric_logger.loss)
            )
        print(
            f"* Macro F1 score: {macro_f1_score:.3f}\n",
            f"* Micro F1 score: {micro_f1_score:.3f}\n",
            f"* Classwise F1 score: {classwise_f1_score}",
        )
    elif args.use_psa:
        # Calculate per-class IoU (Jaccard score)
        iou = jaccard_score(y, predict, average=None)

        # Calculate mean IoU
        miou = np.mean(iou)

        # Log the mIoU
        metric_logger.add_meter("mIoU", misc.AverageMeter())
        metric_logger.update(mIoU=miou)
        print(f"* mIoU: {miou:.3f}\n")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
