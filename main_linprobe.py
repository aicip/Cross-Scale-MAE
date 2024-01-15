# --------------------------------------------------------
# References:
# SatMAE: https://github.com/sustainlab-group/SatMAE
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
import re

from pathlib import Path
from copy import copy
import numpy as np
import glob
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import wandb
from torch.utils.tensorboard import SummaryWriter  # type: ignore

# assert timm.__version__ == "0.3.2"  # version check
import models_vit
import util.misc as misc
from engine_finetune import (
    evaluate,
    train_one_epoch,
)
from timm.models.layers import trunc_normal_
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed
from util.lars import LARS
from util.datasets import build_fmow_dataset


def nullable_string(val):
    if not val:
        return None
    return val


def to_list(x, sep):
    return [int(y) for y in x.split(sep)]


def extract_model_name(dir_path):
    parts = dir_path.split(os.sep)
    if len(parts) >= 2:
        model_part = parts[-2]
        match = re.search(r"out_(.+?_.*?_.*?)(_.*?_|_)", model_part)
        if match:
            model_name = match.group(1)
            next_word = re.search(r"_(.*?)_", match.group(2))
            if next_word and next_word.group(1) not in ["xformers", "scaled"]:
                model_name += next_word.group(1)
            return model_name
    if "mae" in parts[-1]:
        return parts[-1].replace(".pth", "")

    return None


def get_args_parser():
    parser = argparse.ArgumentParser("Cross-MAE Linear Probe", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Batch size per GPU (effective batch size is "
        + "batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        type=int,
        default=1,
        help="Accumulate gradient iterations (for increasing the "
        + "effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=nullable_string,
        default=None,
        choices=["vanilla", None],
        help="Use channel model",
    )
    parser.add_argument(
        "--model",
        default="mae_vit_base",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument(
        "--input_size",
        default=224,
        type=int,
        help="The size of the square-shaped input image",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="The size of the square-shaped patches across the image. "
        + "Must be a divisor of input_size (input_size % patch_size == 0)",
    )
    parser.add_argument(
        "--print_level",
        type=int,
        default=1,
        help="Print Level (0->3) - Only for MaskedAutoencoderShuntedViT",
    )

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay (default: 0 for linear probe following MoCo v1)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help=(
            "Absolute LR. If None, it is set automatically based on absolute_lr ="
            " base_lr * total_batch_size / 256"
        ),
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=0.1,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="Lower LR bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.add_argument(
        "--use_xformers",
        action="store_true",
        help=(
            "Use xFormers instead of Timm for transformer blocks. Not compatible with"
            " --attn_name=shunted"
        ),
    )
    parser.set_defaults(use_xformers=False)
    parser.add_argument(
        "--spatial_mask",
        action="store_true",
        default=False,
        help=(
            "Whether to mask all channels of a spatial location. Only for indp c model"
        ),
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )
    # arg for loss, default is mae
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss function to use",
        choices=[
            "classification_cross",
            "classification_bce",
            "classification_softcross",
        ],
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=False)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument("--use_psa", action="store_true")

    # Dataset parameters
    parser.add_argument(
        "--train_path",
        default="./train_64.csv",
        type=str,
        help="Train .csv path",
    )
    parser.add_argument(
        "--test_path",
        default="/data2/HDD_16TB/fmow-rgb-preproc/val_224.csvv",
        type=str,
        help="Test .csv path",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="rgb",
        choices=[
            "rgb",
            "sentinel",
            "euro_sat",
            "naip",
            "smart",
            "spacenetv1",
            "resisc45",
        ],
        help="Whether to use fmow rgb, sentinel, or other dataset.",
    )
    parser.add_argument(
        "--masked_bands",
        default=None,
        nargs="+",
        type=int,
        help="Sequence of band indices to mask (with mean val) in sentinel dataset",
    )
    parser.add_argument(
        "--dropped_bands",
        type=int,
        nargs="+",
        default=None,
        help="Which bands (0 indexed) to drop from sentinel data.",
    )

    parser.add_argument(
        "--nb_classes", default=62, type=int, help="number of the classification types"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Path used for saving trained model checkpoints and logs. If not specified,"
            " the directory "
        )
        + "name is automatically generated based on model config.",
    )
    parser.add_argument(
        "--output_dir_base",
        type=str,
        default="./out",
        help="Base directory to use for model checkpoints directory",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to use for training / testing",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--resume",
        type=nullable_string,
        default=None,
        help="The path to the checkpoint to resume training from.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="How frequently (in epochs) to save ckpt",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="utk-iccv23",
        help="Wandb entity name, eg: utk-iccv23",
    )
    parser.add_argument(
        "--wandb_project",
        type=nullable_string,
        default=None,
        help="Wandb project name, eg: satmae",
    )
    # https://docs.wandb.ai/guides/runs/resuming
    parser.add_argument(
        "--wandb_id",
        type=nullable_string,
        default=None,
        help="Wandb project id, eg: 83faqrtq",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help=(
            "Enabling distributed evaluation (recommended during training for faster"
            " monitor"
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help=(
            "The number of CPU workers to use for the data loader. Generally, this"
            " should be set to "
        )
        + "the number of CPU threads on your machine.",
    )
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help=(
            "Pin CPU memory in DataLoader for more efficient (sometimes) transfer to"
            " GPU."
        ),
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", 0), type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="URL used to set up distributed training"
    )
    parser.add_argument(
        "--transform_checkpoint_keys",
        action="store_true",
        default=False,
        help=(
            "Whether to attempt to fix errors with loading keys from saved checkpoints."
        ),
    )
    return parser


def main(args):
    misc.init_distributed_mode(args)
    # args.wandb_project = None
    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print("=" * 80)
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    ###########################################################################
    print("=" * 80)
    dataset_train = build_fmow_dataset(is_train=True, args=args)
    dataset_val = build_fmow_dataset(is_train=False, args=args)

    global_rank = misc.get_rank()
    if args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(  # type: ignore
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"Sampler_train = {str(sampler_train)}")
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not"
                    " divisible by process number. This will slightly alter validation"
                    " results as extra duplicate entries are added to achieve equal num"
                    " of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(  # type: ignore
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(  # type: ignore
                dataset_val
            )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # type: ignore
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # type: ignore

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(  # type: ignore
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(  # type: ignore
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    #######################################################################################
    # Define the model
    model = models_vit.__dict__[args.model](
        patch_size=args.patch_size,
        img_size=args.input_size,
        in_chans=dataset_train.in_c,
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]

        # Fix for mapping our models keys to ViT keys
        if not args.transform_checkpoint_keys:
            new_state_dict = checkpoint_model
        else:
            new_state_dict = OrderedDict()
            xformer_mappings = {
                ".wrap_att.norm.": ".norm1.",
                ".wrap_att.sublayer.layer.in_proj_container.q_proj.": ".attn.qkv.",  # splitting here
                ".wrap_att.sublayer.layer.in_proj_container.k_proj.": ".attn.qkv.",
                ".wrap_att.sublayer.layer.in_proj_container.v_proj.": ".attn.qkv.",
                ".wrap_att.sublayer.layer.proj.": ".attn.proj.",
                ".wrap_ff.norm.": ".norm2.",
                ".wrap_ff.sublayer.layer.mlp.0.": ".mlp.fc1.",
                ".wrap_ff.sublayer.layer.mlp.3.": ".mlp.fc2.",
            }
            for key, value in checkpoint_model.items():
                if "encoder" in key:
                    if "encoder_" in key:
                        name = key.replace("encoder_", "")
                    elif "encoder.encoders" in key:  # Xformers case
                        name = key
                        for source, target in xformer_mappings.items():
                            if source in key:
                                name = key.replace(source, target)
                        name = name.replace("encoder.encoders", "blocks")
                    else:
                        name = key.replace("encoder", "blocks")
                    new_state_dict[name] = value
                elif key in {
                    "cls_token",
                    "patch_embed.proj.weight",
                    "patch_embed.proj.bias",
                }:
                    name = key
                    new_state_dict[name] = value

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)

        # Print the keys that were actually loaded
        pretrained_keys = set(new_state_dict.keys())
        current_keys = set(model.state_dict().keys())
        loaded_keys = pretrained_keys.intersection(current_keys)
        loaded_keys = sorted(list(loaded_keys))
        grouped_keys = {}
        for key in loaded_keys:
            main_key = key.split(".")[0]
            if main_key not in grouped_keys:
                grouped_keys[main_key] = []
            grouped_keys[main_key].append(key)
        print("Keys that were actually loaded:")
        for main_key, sub_keys in grouped_keys.items():
            print(f"{main_key}: {len(sub_keys)} weight(s)")

        if args.global_pool:
            assert set(msg.missing_keys) == {
                "head.weight",
                "head.bias",
                "fc_norm.weight",
                "fc_norm.bias",
            }
        else:
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

        model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head,
        )
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    model = model.to(device)

    model_without_ddp = model
    print(f"Model = {str(model_without_ddp)}")

    ###########################################################################
    print("=" * 80)
    batch_size_eff = args.batch_size * args.accum_iter * misc.get_world_size()

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % batch_size_eff)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_size_eff / 256

    print("base lr: %.2e" % (args.lr * 256 / batch_size_eff))
    print("actual lr: %.2e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    ###########################################################################
    print("=" * 80)
    # build optimizer with layer-wise lr decay (lrd)

    param_groups = model_without_ddp.head.parameters()
    optimizer = LARS(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.loss == "classification_cross":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise Exception("Only classification_cross is supported")

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        device=args.device,
    )

    print("=" * 80)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_num_params = sum(np.prod(p.size()) for p in model_params)
    print(f"Trainable parameters: {model_num_params}")

    #######################################################################################

    model_name: str = "_".join(
        [
            args.model,
            f"i{args.input_size}-p{args.patch_size}",
            f"e{args.epochs}-we{args.warmup_epochs}",
            f"b{args.batch_size}-a{args.accum_iter}",
            f"{args.loss}{'-normpix' if args.norm_pix_loss else ''}",
            f"-lr{args.lr}",
            "_cls_only" if not args.global_pool else "_global_pool",
            f"CHKP{extract_model_name(args.finetune)}-linprobe",
        ]
    )

    if args.output_dir is None:
        args.output_dir = f"out_{model_name}"
    if args.output_dir_base is not None:
        args.output_dir = os.path.join(args.output_dir_base, args.output_dir)

    # finding a new output directory if one already exists with the same name
    if args.resume is None:
        while os.path.exists(args.output_dir):
            # print out if doesn't have any .pth files
            if len(glob.glob(os.path.join(args.output_dir, "*.pth"))) == 0:
                print(
                    f"INFO: {args.output_dir} already exists, but contains no .pth"
                    " files. You may want to delete it."
                )
            number = os.path.basename(args.output_dir).split("+")[-1]
            number = int(number) + 1 if number.isdigit() else 1
            args.output_dir = os.path.join(
                os.path.dirname(args.output_dir), f"out_{model_name}+{number}"
            )

    print(f"Output directory: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    log_writer = None
    if misc.is_main_process():
        if args.wandb_entity is not None and args.wandb_project is not None:
            wandb_id = (
                wandb.util.generate_id()
                if args.resume is None  # type: ignore
                else args.wandb_id
            )
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=model_name,
                group=args.model,
                job_type="linprobe",
                resume=None if args.resume is None else "must",
                id=wandb_id,
            )
            wandb_args = copy(args)
            if args.resume is not None:
                wandb_args.start_epoch += 1
            wandb.config.update(wandb_args, allow_val_change=True)
            wandb.config.update(
                {"num_params": model_num_params, "batch_size_eff": batch_size_eff}
            )
            wandb.watch(model)
        else:
            print("INFO: Not using WandB.")

        # Logging
        if args.output_dir is not None:
            # output_dir_tb = os.path.join(args.output_dir, "tensorboard")
            output_dir_tb = os.path.join("./logs")
            log_writer = SummaryWriter(log_dir=output_dir_tb)
            print(f"INFO: Tensorboard log path: {output_dir_tb}")
        else:
            print("INFO: Not logging to tensorboard.")

    ###########################################################################

    if args.eval:
        test_stats = evaluate(
            data_loader_val, model, device, ignore_index=dataset_val.ignore_index
        )
        print(
            f"Evaluation on {len(dataset_val)} test images:"
            f"\n\tacc1: {test_stats['acc1']:.2f}%"
            f"\n\tacc5: {test_stats['acc5']:.2f}%, "
            f"\n\tmacro_f1: {test_stats['macro_f1']:.2f}%, "
            f"\n\tmicro_f1: {test_stats['micro_f1']:.2f}%"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=None,
            mixup_fn=None,
            log_writer=log_writer,
            args=args,
            ignore_index=dataset_train.ignore_index,
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and (
            (epoch % args.save_every == 0 and epoch >= args.epochs / 2)
            or (epoch % 5 == 0 and epoch < args.epochs / 2)
            or epoch + 1 == args.epochs
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        test_stats = evaluate(data_loader_val, model, device, args)

        print(
            f"Accuracy of the network on the {len(dataset_val)} test images:"
            f" {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
            log_writer.add_scalar("perf/test_macro_f1", test_stats["macro_f1"], epoch)
            log_writer.add_scalar("perf/test_micro_f1", test_stats["micro_f1"], epoch)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.jsonl"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
