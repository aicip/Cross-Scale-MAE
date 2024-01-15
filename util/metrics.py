import torch
from pytorch_msssim import ms_ssim, ssim


def calc_ssim(x, y, num_channels=3):
    # convert to [N, C, H, W]
    if x.shape[-1] == num_channels:
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
    return ssim(x, y, data_range=1, size_average=True).item()


METRICS_DICT = {
    "mse": {
        "full_name": "Mean Squared Error",
        "is_lower_better": True,
        "lambda": lambda x, y: torch.mean((x - y) ** 2).item(),
    },
    "mae": {
        "full_name": "Mean Absolute Error",
        "is_lower_better": True,
        "lambda": lambda x, y: torch.mean(torch.abs(x - y)).item(),
    },
    "l1": {
        "full_name": "L1 Norm",
        "is_lower_better": True,
        "lambda": lambda x, y: torch.sum(torch.abs(x - y)).item(),
    },
    "l2": {
        "full_name": "L2 Norm",
        "is_lower_better": True,
        "lambda": lambda x, y: torch.sum((x - y) ** 2).item(),
    },
    "ssim": {
        "full_name": "Structural Similarity Index",
        "is_lower_better": False,
        "lambda": calc_ssim,
    },
    # Note: MS-SSIM requires image size to be larger than 160
    # AssertionError: Image size should be larger than 160 due to the 4 downsamplings in ms-ssim
    "ms_ssim": {
        "full_name": "Multi-Scale Structural Similarity Index",
        "is_lower_better": False,
        "lambda": lambda x, y: ms_ssim(x, y, data_range=1, size_average=True).item(),
    },
}


def calc_metric(x, y, metric_name):
    comp_metric = metric_name.lower()
    if comp_metric == "ssd":
        comp_metric = "l2"
    elif comp_metric == "sad":
        comp_metric = "l1"

    return METRICS_DICT[comp_metric]["lambda"](x, y)
