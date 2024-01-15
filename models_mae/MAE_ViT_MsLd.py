import torch
import torch.nn as nn
from torchvision import transforms as T

from .MAE_ViT_Baseline import MAE_ViT_Baseline


class MAE_ViT_MsLd(MAE_ViT_Baseline):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        # Range of random crop for Multi-scale training
        ms_range=(0.25, 0.75),
        # Reduction for decoder losses between original and cropped images
        ms_decoder_loss_reduction: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ms_decoder_loss_reduction = ms_decoder_loss_reduction.lower()

        self.allowed_reductions = ["mean", "sum"]
        assert (
            self.ms_decoder_loss_reduction in self.allowed_reductions
        ), f"ms_decoder_loss_reduction must be one of: {self.allowed_reductions}"

        print(f"ms_decoder_loss_reduction: {self.ms_decoder_loss_reduction}")

        self.crop = nn.Sequential(
            T.RandomResizedCrop(
                size=(self.input_size, self.input_size),
                scale=ms_range,
                antialias=True,
            )
        )

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False
    ):
        if mask_seed is not None:
            torch.manual_seed(mask_seed)
        elif consistent_mask:
            # Make sure the mask is consistent for all images in the batch
            mask_seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Random crop image
        imgs_crop = self.crop(imgs)

        # Forward Original image
        loss_orig, pred_orig, mask_orig, enc_emb_orig, dec_emb_orig = super().forward(
            imgs, mask_ratio=mask_ratio, mask_seed=mask_seed, return_embeds=True
        )
        # Forward Cropped image
        loss_crop, pred_crop, mask_crop, enc_emb_crop, dec_emb_crop = super().forward(
            imgs_crop, mask_ratio=mask_ratio, mask_seed=mask_seed, return_embeds=True
        )

        # Reconstruction loss combining original and cropped image
        loss_d = loss_orig + loss_crop
        if self.ms_decoder_loss_reduction == "mean":
            loss_d /= 2

        if not return_embeds:
            return loss_d, pred_orig, mask_orig

        return (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )

class MAE_ViT_MsLd_PAIRED(MAE_ViT_Baseline):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        # Range of random crop for Multi-scale training
        ms_range=(0.2, 0.8),
        # Reduction for decoder losses between original and cropped images
        ms_decoder_loss_reduction: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ms_decoder_loss_reduction = ms_decoder_loss_reduction.lower()

        self.allowed_reductions = ["mean", "sum"]
        assert (
            self.ms_decoder_loss_reduction in self.allowed_reductions
        ), f"ms_decoder_loss_reduction must be one of: {self.allowed_reductions}"

        print(f"ms_decoder_loss_reduction: {self.ms_decoder_loss_reduction}")

        # self.crop = nn.Sequential(
        #     T.RandomResizedCrop(
        #         size=(self.input_size, self.input_size),
        #         scale=ms_range,
        #         antialias=True,
        #     )
        # )

    def forward(
        self,
        imgs1,
        imgs2,
        mask_ratio=0.75,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False
    ):
        if mask_seed is not None:
            torch.manual_seed(mask_seed)
        elif consistent_mask:
            # Make sure the mask is consistent for all images in the batch
            mask_seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Forward Original image
        loss_orig, pred_orig, mask_orig, enc_emb_orig, dec_emb_orig = super().forward(
            imgs1, mask_ratio=mask_ratio, mask_seed=mask_seed, return_embeds=True
        )
        # Forward Cropped image
        loss_crop, pred_crop, mask_crop, enc_emb_crop, dec_emb_crop = super().forward(
            imgs2, mask_ratio=mask_ratio, mask_seed=mask_seed, return_embeds=True
        )

        # Reconstruction loss combining original and cropped image
        loss_d = loss_orig + loss_crop
        if self.ms_decoder_loss_reduction == "mean":
            loss_d /= 2

        if not return_embeds:
            return loss_d, pred_orig, mask_orig

        return (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )
