import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from tqdm import tqdm

import models_mae
import util.metrics as metrics
from util.misc import (
    title_to_fname,
    print_checkpoint_folders,
    glob_helper,
    seed_str_to_int,
)

# For normalizing image inputs based on the dataset's mean and std
# TODO: replace these hardcoded values with the ones referenced from dataset
image_mean = np.array([0.40558367, 0.43378946, 0.43175863])
image_std = np.array([0.19208308, 0.19136319, 0.19783947])


def prepare_model(
    chkpt_dir,
    chkpt_basedir="../Model_Saving",
    chkpt_name=None,
    map_location="cpu",
):
    """
    Loads the model from checkpoint

    Arguments:
        chkpt_dir -- Ex: out_mae_vit_small_scaled_dot_product_i128_p16_b512_e200

    Keyword Arguments:
        chkpt_basedir -- Base directory where multiple models checkpoints are stored directories (default: {"../Model_Saving"})
        chkpt_name -- If none, load the latest (default: {None})
        arch -- Model architecture being used (default: {"mae_vit_base"})
        map_location -- Where to load the state dict of model (default: {"cpu"})

    Returns:
        The model
    """
    print("=" * 80)
    checkpoint_folder = os.path.join(chkpt_basedir, chkpt_dir)

    try:
        if chkpt_name is None:
            # List the directory and find the checkpoint with the highest epoch
            checkpoint_list = os.listdir(checkpoint_folder)
            checkpoint_list = [x for x in checkpoint_list if x.endswith(".pth")]
            checkpoint_list.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            chkpt_name = checkpoint_list[-1]
    except (FileNotFoundError, IndexError) as e:
        print("Could not find: ", checkpoint_folder)
        print_checkpoint_folders(chkpt_basedir)
        raise e

    if not chkpt_name.endswith(".pth"):
        chkpt_name = f"{chkpt_name}.pth"

    if not chkpt_name.startswith("checkpoint-"):
        chkpt_name = f"checkpoint-{chkpt_name}"

    chkpt_path = os.path.join(checkpoint_folder, chkpt_name)
    print("Loading checkpoint: ", chkpt_path)
    checkpoint = torch.load(chkpt_path, map_location=map_location)

    # build model
    args = vars(checkpoint["args"])
    if "print_level" in args:
        args["print_level"] = 0
    print("args:", args)
    try:
        model = getattr(models_mae, args["model"])(**args)
    except AssertionError as e:
        print("Error: ", e)
        return None
    # load model
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    if model.device is not None:
        model = model.to(model.device)
    print(msg)
    print("Model loaded.")
    return model


def prepare_image(
    image_uri, img_size, random_crop=False, crop_seed=None, resample=None, **kwargs
):
    """
    :param resample: An optional resampling filter.  This can be
           one `Resampling.NEAREST`, `Resampling.BOX`,
           `Resampling.BILINEAR`, `Resampling.HAMMING`,
           `Resampling.BICUBIC`, `Resampling.LANCZOS`.
    """
    img = Image.open(image_uri)

    # random crop
    if random_crop:
        if crop_seed is not None:
            torch.manual_seed(crop_seed)

        interpol_mode = T.InterpolationMode.BICUBIC
        img = T.RandomResizedCrop(
            img_size,
            scale=(0.25, 1.0),
            interpolation=interpol_mode,
            antialias=True,
        )(img)

    img = img.resize((img_size, img_size), resample=resample)
    img = np.array(img) / 255.0
    img = (img - image_mean) / image_std

    return img


def add_noise(image, noise_type="gaussian", noise_param=0.1):
    # if not tensor, convert to tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    # create a tensor the same size as the image
    if noise_type == "gaussian":
        noise = torch.randn_like(image) * noise_param
    elif noise_type == "poisson":
        noise = torch.poisson(torch.ones_like(image) * noise_param)
    elif noise_type == "s&p":
        noise = torch.bernoulli(torch.ones_like(image) * noise_param)
    else:
        raise ValueError("Invalid noise type")

    return image + noise.to(image.device)


@torch.no_grad()
def run_one_image(
    img,
    model,
    mask_seed: Optional[int] = None,
    **kwargs,
):
    if "patch_size" not in model.__dict__:  # for shunted models
        patch_size = model.patch_sizes[-1]
    else:
        patch_size = model.patch_size
    channels = model.input_channels

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)

    # run MAE
    if "mask_ratio" in model.__dict__:
        mask_ratio = model.mask_ratio
    else:
        mask_ratio = 0.75
        print(
            f"WARN: mask_ratio not found in model config. Defaulting to {mask_ratio}."
        )

    xf = x.float()
    if model.device is not None:
        xf = xf.to(model.device)

    _, y, mask = model(xf, mask_ratio=mask_ratio, mask_seed=mask_seed)

    y = model.unpatchify(y, p=patch_size, c=channels)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    if "num_stages" in model.__dict__:
        stage = model.num_stages - 1
        patch_embed = getattr(model, f"patch_embed{stage + 1}")
    else:
        patch_embed = model.patch_embed
    mask = mask.unsqueeze(-1).repeat(
        1, 1, patch_embed.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(
        mask, p=patch_size, c=channels
    )  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    # Convert to channel last
    x = torch.einsum("nchw->nhwc", x)

    # Un-normalize
    x = (x * image_std) + image_mean
    y = (y * image_std) + image_mean

    # masked image
    xm = x * (1 - mask)

    ym = y * mask
    # MAE reconstruction pasted with visible patches
    xm_ym = xm + ym

    return x, xm, y, ym, xm_ym


def plot_image(image, ax=None, title="", figsize=4, show=False):
    # if first dimension is 1, squeeze batch dimension
    if image.shape[0] == 1:
        image = image.squeeze(dim=0)

    assert len(image.shape) == 3, "image should be (H, W, C)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize, figsize))

    # if needed conver to int 0-255
    if image.dtype != np.uint8:
        image = torch.clip((image) * 255, 0, 255).int()

    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    if show:
        plt.show()


def plot_reconstruction(
    models: Union[dict, torch.nn.Module],
    image,
    image_name: Optional[str] = None,
    comp_metric: str = "ssim",
    title: Optional[str] = None,
    figsize: int = 10,
    savedir: str = "./plots/",
    save: bool = False,
    show: bool = True,
    **kwargs,
):
    """
    :param resample: An optional resampling filter.  This can be
           one `Resampling.NEAREST`, `Resampling.BOX`,
           `Resampling.BILINEAR`, `Resampling.HAMMING`,
           `Resampling.BICUBIC`, `Resampling.LANCZOS`.
    """
    # if model is not an array, make it an array
    if not isinstance(models, dict):
        models = {"model": models}

    plt.clf()
    num_cols = 3

    fig, axs = plt.subplots(
        len(models),
        num_cols,
        figsize=(figsize, len(models) * figsize / 3.0),
    )

    savesubdir = None
    if title is not None:
        if image_name is not None:
            savesubdir = title_to_fname(title)
            title = f"{title} - {image_name}"
        fig.suptitle(title)

    for model_i, (model_name, model) in enumerate(models.items()):
        if isinstance(image, str):
            img = prepare_image(image, model.input_size, **kwargs)
        else:
            img = image.copy()

        x, xm, y, ym, xm_ym = run_one_image(img, model, **kwargs)

        diff = np.abs(x - y)
        diff_m = metrics.calc_metric(x, y, comp_metric)
        # diff_m = metrics.calc_metric(x, xm_ym, comp_metric)

        imgs_dict = {
            "Original": x,
            "Input (Masked)": xm,
            f"{model_name} ({comp_metric.upper()}: {diff_m:<.3f})": y,
            # f"{comp_metric.upper()}: {diff_m:<.3f}": diff,
            # f"Paste Visible ({comp_metric.upper()}: {diff_m:<.3f})": xm_ym,
            # "Reconstruction + Visible": xm_ym,
        }

        for i, (ti, im) in enumerate(imgs_dict.items()):
            ax = axs[model_i, i] if len(models) > 1 else axs[i]
            plot_image(im, ax, ti)

    plt.tight_layout()
    if save:
        if title is not None:
            save_fname = title_to_fname(title)
            if savesubdir is not None:
                savedir = os.path.join(savedir, savesubdir)
            os.makedirs(savedir, exist_ok=True)
            save_path = os.path.join(savedir, f"plot_img_{save_fname}.png")
            plt.savefig(save_path)
        else:
            print("INFO: Skipped saving because title was not provided")

    if show:
        plt.show()

    # Return the plot as a pixel array
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def run_eval(
    models: dict,
    basedir: str,
    comp_metrics: Optional[
        Union[str, List[str]]
    ] = None,  # None defaults to all metrics
    use_noise: Optional[tuple] = None,  # ex: ("gaussian", 0.25)
    num_runs_each: int = 5,
    do_plot_metrics_comp=True,
    do_plot_image_comp=False,
    plot_every: int = None,
    **kwargs,
):
    if comp_metrics is not None:
        comp_metrics_list = (
            [comp_metrics] if isinstance(comp_metrics, str) else comp_metrics
        )
        image_comp_metric = comp_metrics_list[0]
    else:
        comp_metrics_list = list(metrics.METRICS_DICT.keys())
        # TODO: MS_SSIM needs image size to be larger than 160
        comp_metrics_list.remove("ms_ssim")
        image_comp_metric = "ssim"

    def mk_subtitle():
        t = f"Avg of {num_runs_each} runs/sample"
        if kwargs.get("random_crop", False):
            t += " w/ random crop"
        if use_noise is not None:
            t += f" + {use_noise[0]} noise {use_noise[1]}"
        return t

    # dictionary for metric for each model
    mtrs = {
        metric: {model_name: [] for model_name in models}
        for metric in comp_metrics_list
    }

    image_comp_metric_is_lower_better = metrics.METRICS_DICT[image_comp_metric][
        "is_lower_better"
    ]

    img_i = 0
    biggest_gap = None
    # TODO: This can definitely be made faster with the following:
    # E.g. running whole batches and on GPU rather than one image at a time on CPU
    for img_i, img_path in tqdm(
        enumerate(glob_helper(f"{basedir}/**/*.jpg", **kwargs))
    ):
        best_model = None
        worst_model = None
        for model_name, model in models.items():
            mtrs_runavg = {metric: 0 for metric in comp_metrics_list}

            best_run = None
            for run_i in range(num_runs_each):
                img = prepare_image(
                    img_path,
                    model.input_size,
                    # Crop seed used to keep crop consistent across models, but different for each image, and different for each run
                    crop_seed=seed_str_to_int(f"{img_i}-{run_i}"),
                    **kwargs,
                )

                # TODO: Noise may need to be applied before random crop
                if use_noise is not None:
                    img = add_noise(
                        img, noise_type=use_noise[0], noise_param=use_noise[1]
                    )

                mask_seed = seed_str_to_int(f"{img_i}-{run_i}")
                x, xm, y, ym, xm_ym = run_one_image(
                    img,
                    model,
                    mask_seed=mask_seed,
                    **kwargs,
                )

                for metric_name in comp_metrics_list:
                    metric_val = metrics.calc_metric(x, y, metric_name)
                    mtrs_runavg[metric_name] += metric_val

                image_comp_metric_val = metrics.calc_metric(x, y, image_comp_metric)

                # For debugging to make sure consistent masks and crops are used across models
                # plot_image(xm, title=f"{img_path} - {model_name} - {run_i} - {mask_seed} - {image_comp_metric}: {image_comp_metric_val:.3f}", show=True)

                if (
                    best_run is None
                    or (
                        image_comp_metric_is_lower_better
                        and image_comp_metric_val < best_run[0]
                    )
                    or (
                        not image_comp_metric_is_lower_better
                        and image_comp_metric_val > best_run[0]
                    )
                ):
                    best_run = (image_comp_metric_val, img, mask_seed)

            # the best model
            if (
                best_model is None
                or (image_comp_metric_is_lower_better and best_run[0] < best_model[0])
                or (
                    not image_comp_metric_is_lower_better
                    and best_run[0] > best_model[0]
                )
            ):
                best_model = best_run

            # the worst model
            if (
                worst_model is None
                or (image_comp_metric_is_lower_better and best_run[0] > worst_model[0])
                or (
                    not image_comp_metric_is_lower_better
                    and best_run[0] < worst_model[0]
                )
            ):
                worst_model = best_run

            for metric_name in comp_metrics_list:
                mtrs[metric_name][model_name].append(
                    mtrs_runavg[metric_name] / num_runs_each
                )

        plotted_image = False
        plotted_metrics = False

        if plot_every is not None and img_i % plot_every == 0:
            if do_plot_image_comp:
                plot_reconstruction(
                    models,
                    best_model[1],
                    image_name=os.path.basename(img_path).split(".")[0],
                    mask_seed=best_model[2],
                    **kwargs,
                )
                plotted_image = True

            if do_plot_metrics_comp:
                kwargs["subtitle"] = mk_subtitle()
                plot_metrics_comp(mtrs, show=False, **kwargs)
                plotted_metrics = True

        difference = np.abs(best_model[0] - worst_model[0])

        # Plot 'notable' examples (where there is a large gap between best and worst model for a given image)
        if (
            biggest_gap is None
            or difference > biggest_gap[0] * 0.85
            and biggest_gap[0] != difference
        ):
            biggest_gap = (difference, best_model[1], best_model[2])

            if not plotted_image and do_plot_image_comp:
                plot_reconstruction(
                    models,
                    biggest_gap[1],
                    image_name=os.path.basename(img_path).split(".")[0] + "*",
                    mask_seed=biggest_gap[2],
                    **kwargs,
                )

            if not plotted_metrics and do_plot_metrics_comp:
                kwargs["subtitle"] = mk_subtitle()
                # plot_metrics_comp(mtrs, show=False, **kwargs)
                plot_metrics_comp(mtrs, **kwargs)

    print(
        f"# Finished evaluating on: {basedir} - {img_i + 1} images for {len(models)} models ({num_runs_each} runs each)"
    )

    kwargs["subtitle"] = mk_subtitle()

    if do_plot_metrics_comp:
        plot_metrics_comp(mtrs, **kwargs)

    return mtrs


def plot_metrics_comp(
    metrics_dict: dict,
    figsize: tuple = (2.5, 3),
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    type: str = "line",
    save=False,
    savedir="./plots",
    show=True,
    **kwargs,
):
    # show one row for each metric
    num_rows = len(metrics_dict)
    # get the number of models
    num_cols = len(list(metrics_dict.values())[0])
    # clear fig
    plt.clf()
    fig, axs = plt.subplots(
        num_rows, 1, figsize=(figsize[0] * num_cols, figsize[1] * num_rows)
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.05, hspace=0.3)

    if title is not None:
        fig.suptitle(title)

    if subtitle is None:
        subtitle = f""

    fig.text(0.5, 0.92, subtitle, ha="center", va="center")

    for mi, (metric_name, metric_d) in enumerate(metrics_dict.items()):
        ax = axs[mi] if num_rows > 1 else axs
        # type is line, bar, box
        is_lower_better = metrics.METRICS_DICT[metric_name]["is_lower_better"]
        if type == "bar":
            vals = []
            for model_name, model_metric in metric_d.items():
                mean = np.mean(model_metric)
                std = np.std(model_metric)
                vals.append((model_name, mean, std))

            vals_sorted = sorted(vals, key=lambda x: x[1], reverse=not is_lower_better)

            # for i, (model_name, mean, std) in enumerate(
            #     vals_sorted
            # ):
            best_model = vals_sorted[0][0]
            worst_model = vals_sorted[-1][0]

            for i, (model_name, mean, std) in enumerate(vals):
                ax.bar(
                    model_name,
                    mean,
                    yerr=std,
                    label=f"{mean:.3f} ± {std:.3f}",
                    # color the error bar red
                    error_kw=dict(ecolor="red", lw=1, capsize=3, capthick=1),
                )

                # green if the mean is the best, red if the mean is the worst, gray otherwise
                color = (
                    "lime"
                    if model_name == best_model
                    else "red"
                    if model_name == worst_model
                    else "slategray"
                )

                ax.axhline(
                    mean,
                    color=color,
                    linestyle="dashdot",
                    linewidth=1,
                    alpha=1.0,
                )

                # show the mean and std at the base of the bar
                ax.text(
                    model_name,
                    ax.get_yticks()[1] * 0.25,
                    f"{mean:.3f} ± {std:.3f}",
                    ha="center",
                    va="bottom",
                    color="black",
                )

        elif type == "box":
            ax.boxplot(
                metric_d.values(),
                labels=metric_d.keys(),
                showmeans=True,
                meanline=True,
                meanprops=dict(linestyle="--", color="red"),
            )
            ax.legend(framealpha=0.25)

        elif type == "line":
            for model_name, model_metric in metric_d.items():
                ax.plot(model_metric, label=model_name)
                ax.set_xticks(range(len(model_metric)))
        else:
            raise ValueError(f"Invalid type: {type}")

        ax.set_title(
            f"{metrics.METRICS_DICT[metric_name]['full_name']} ({metric_name.upper()}) - {'Lower' if is_lower_better else 'Higher'} is better"
        )

    if type == "line":
        handles, labels = axs[0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels))

    if save:
        if title is not None:
            save_fname = title_to_fname(f"{title} - {subtitle}")
            os.makedirs(savedir, exist_ok=True)
            plt.savefig(os.path.join(savedir, f"plot_{type}_{save_fname}.png"))
        else:
            print("INFO: Skipped saving because title was not provided")

    if show:
        plt.show()
