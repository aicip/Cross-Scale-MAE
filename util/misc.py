# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import glob
import os
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

# from torch._six import inf
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

        self.MB = 1024.0 * 1024.0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = [f"{name}: {str(meter)}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = f":{len(str(len(iterable)))}d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "iter_time: {iter_time}",
            "data_time: {data_time}",
        ]
        if torch.cuda.is_available():
            log_msg.append("memory: {memory:.0f}")

        log_msg = self.delimiter.join(log_msg)

        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    memory_alloc = torch.cuda.max_memory_allocated() / self.MB
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_str,
                            meters=str(self),
                            iter_time=str(iter_time),
                            data_time=str(data_time),
                            memory=memory_alloc,
                        )
                    )
                    self.update(memory_alloc=memory_alloc)
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_str,
                            meters=str(self),
                            iter_time=str(iter_time),
                            data_time=str(data_time),
                        )
                    )
            end = time.time()
        # Seconds
        total_time = time.time() - start_time
        total_time_per_it = total_time / len(iterable)

        self.update(time_epoch=total_time, time_step=total_time_per_it)

        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time_per_it
            )
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.global_sum = 0
        self.global_count = 0
        self.global_avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.global_sum += val * n
        self.global_count += n
        self.global_avg = self.global_sum / self.global_count


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    return bool(dist.is_initialized()) if dist.is_available() else False


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}, world size {args.world_size}",
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / f"checkpoint-{epoch_name}.pth"]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag=f"checkpoint-{epoch_name}",
            client_state=client_state,
        )


def load_model(args, model_without_ddp, optimizer, loss_scaler, device=None):
    print("resume is ****")
    print(type(args.resume))
    print(args.resume)
    if not args.resume or args.resume is None:
        print("Not resuming from checkpoint")
        return

    checkpoint = (
        torch.hub.load_state_dict_from_url(
            args.resume, map_location="cpu", check_hash=True
        )
        if args.resume.startswith("https")
        else torch.load(args.resume, map_location="cpu")
    )
    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    if device is not None:
        model_without_ddp.to(device)  # Move the model to the correct device
        
    print(f"Resuming from checkpoint: {args.resume}")
    if (
        "optimizer" in checkpoint
        and "epoch" in checkpoint
        and not (hasattr(args, "eval") and args.eval)
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size <= 1:
        return x
    x_reduce = torch.tensor(x).cuda()
    dist.all_reduce(x_reduce)
    x_reduce /= world_size
    return x_reduce.item()


def squeeze_many(*args):
    return [torch.squeeze(x) for x in args]


def title_to_fname(title):
    # replace symbols, spaces, etc to make a nice-looking filename
    save_fname = title.replace("-", "")
    save_fname = re.sub(r"[^\w\s]", "_", save_fname)
    save_fname = re.sub(r"\s+", "_", save_fname)
    while "__" in save_fname:
        save_fname = save_fname.replace("__", "_")
    save_fname = save_fname.strip("_")
    return save_fname


def seed_str_to_int(seed_str):
    # convert seed string to int using ord of each char
    seed_int = 0
    for c in seed_str:
        seed_int += ord(c)
    return seed_int


def print_checkpoint_folders(chkpt_basedir):
    print("Available checkpoint folders:")
    # print only folders that contain .pth files
    potential_folders = []
    for folder in glob.glob(f"{chkpt_basedir}/**/*", recursive=True):
        if len(glob.glob(f"{folder}/*.pth")) > 0:
            # also print time last modified in time ago format
            last_modified = os.path.getmtime(folder)
            time_agos = [
                "sec",
                "min",
                "hrs",
                "days",
                "wks",
                "mts",
                "yrs",
            ]
            potential_folders.append((folder, last_modified))

    potential_folders.sort(key=lambda x: x[1], reverse=True)

    for folder, last_modified in potential_folders:
        last_modified = time.time() - last_modified
        time_ago = "some time"
        for time_ago_ in time_agos:  # type: ignore
            time_ago = time_ago_
            if last_modified < 60:
                break
            last_modified = last_modified / 60

        last_modified = f"{last_modified:.1f} {time_ago} ago"
        folderpath_clean = folder.replace(chkpt_basedir, "").strip("/")
        print(f" - {folderpath_clean:<100} ({last_modified})")


def glob_helper(
    glob_pattern: str,
    max_samples: Optional[int] = None,
    random_walk: bool = False,  # If true, must specify max_samples
    walk_seed: Optional[int] = None,  # Only used if random_walk is True
    **kwargs,
):
    # if random_walk then max_samples should not be None
    assert not (
        random_walk and max_samples is None
    ), "must specify max_samples if random_walk is True"

    # if walkseed is not None, then random_walk should be True
    assert not (
        walk_seed is not None and not random_walk
    ), "walkseed can only be used if random_walk is True"

    if walk_seed is not None:
        random.seed(walk_seed)

    if random_walk:
        # get all files matching glob pattern
        all_files = glob.glob(glob_pattern, recursive=True)

        # get random sample of max_samples files
        sample = random.sample(all_files, max_samples)

        # yield each file in sample
        for file in sample:
            yield file
    else:
        for i, img_path in enumerate(glob.iglob(glob_pattern, recursive=True)):
            if max_samples is not None and i >= max_samples:
                break
            yield img_path
