import abc
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim, ssim
from timm.loss import SoftTargetCrossEntropy  # ,LabelSmoothingCrossEntropy


class MAE_ViT_Shared(nn.Module):
    def __init__(
        self,
        norm_pix_loss=False,
        loss="mse",
        **kwargs,
    ):
        super().__init__()
        self.loss = loss.lower()
        self.norm_pix_loss = norm_pix_loss
        # get the loss function from class based on string
        self.__forward_loss = getattr(self, f"forward_loss_{self.loss}")
        print(
            f"__forward_loss: {self.loss} -> {self.__forward_loss.__name__} (norm_pix_loss={self.norm_pix_loss})"
        )

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    @abc.abstractmethod
    def forward_encoder(self, x, mask_ratio):
        pass

    @abc.abstractmethod
    def forward_decoder(self, x, ids_restore):
        pass

    def scale_01(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1.0e-6)

    def process_target(self, imgs, patch_embed_psize, input_channels):
        """
        imgs (before): [N, 3, H, W]
        target (after): [N, L, p*p*3]
        """
        target = self.patchify(imgs, patch_embed_psize, input_channels)
        # print("target", target.shape)
        # torch.Size([512, 64, 192])

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        return target

    def forward_loss_mse(self, target, pred, mask=None, **kwargs):
        loss = (pred - target) ** 2  # torch.Size([512, 64, 192])
        # loss per patch [N, L]
        loss = loss.mean(dim=-1)  # torch.Size([512, 64])

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_l2(self, target, pred, mask=None, **kwargs):
        loss = (pred - target) ** 2  # torch.Size([512, 64, 192])
        # loss per patch [N, L]
        loss = loss.sum(dim=-1)  # torch.Size([512, 64])

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_mae(self, target, pred, mask=None, **kwargs):
        loss = torch.abs(pred - target)  # torch.Size([512, 64, 192])
        # loss per patch [N, L]
        loss = loss.mean(dim=-1)  # torch.Size([512, 64])

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_l1(self, target, pred, mask=None, **kwargs):
        loss = torch.abs(pred - target)  # torch.Size([512, 64, 192])
        # loss per patch [N, L]
        loss = loss.sum(dim=-1)  # torch.Size([512, 64])

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_bce(self, target, pred, mask=None, **kwargs):
        # From Docs:
        # input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
        # target: Tensor of the same shape as input with values between 0 and 1
        target = self.scale_01(target)

        loss = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        # loss per patch [N, L]
        loss = loss.mean(dim=-1)

        # mask: [N, L], 0 is visible, 1 is reconstructed,
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward_loss_ssim(
        self, target, pred, mask=None, patch_embed_psize=None, input_channels=None
    ):
        """
        From Docs: https://github.com/VainF/pytorch-msssim
        "If you need to calculate MS-SSIM/SSIM on normalized images,
        please denormalize them to the range of [0, 1] or [0, 255] first.
        For ssim, it is recommended to set nonnegative_ssim=True to avoid negative results.
        However, this option is set to False by default to keep it consistent with tensorflow and skimage.
        For ms-ssim, there is no nonnegative_ssim option and the ssim reponses
        is forced to be non-negative to avoid NaN results."
        """
        # Pred, Target: [N, L, p*p*3]
        # Mask: [N, L]

        # SSIM and MS-SSIM functions require input to be in range [0, 1]
        target, pred = self.scale_01(target), self.scale_01(pred)

        # SSIM and MS-SSIM functions require input [N, C, H, W]
        target = self.unpatchify(target, patch_embed_psize, input_channels)
        pred = self.unpatchify(pred, patch_embed_psize, input_channels)

        # By default perform SSIM on reconstructed masked patches only (when used stand-alone)
        # Optionally, if mask is None, perform SSIM on all patches (reconstructed and visible)
        if mask is not None:
            if patch_embed_psize is None or input_channels is None:
                raise ValueError(
                    "patch_embed_psize and input_channels must be provided if mask is provided"
                )
            mask = mask.unsqueeze(-1).repeat(
                1, 1, patch_embed_psize**2 * 3
            )  # (N, H*W, p*p*3)
            mask = self.unpatchify(
                mask, p=patch_embed_psize, c=input_channels
            )  # 1 is removing, 0 is keeping

            target = target * mask
            pred = pred * mask

        return 1 - ssim(
            pred, target, data_range=1, size_average=True, nonnegative_ssim=True
        )

    def forward_loss_ms_ssim(
        self, target, pred, mask=None, patch_embed_psize=None, input_channels=None
    ):
        """
        From Docs: https://github.com/VainF/pytorch-msssim
        "If you need to calculate MS-SSIM/SSIM on normalized images,
        please denormalize them to the range of [0, 1] or [0, 255] first.
        For ssim, it is recommended to set nonnegative_ssim=True to avoid negative results.
        However, this option is set to False by default to keep it consistent with tensorflow and skimage.
        For ms-ssim, there is no nonnegative_ssim option and the ssim reponses
        is forced to be non-negative to avoid NaN results."
        """
        # Pred, Target: [N, L, p*p*3]
        # Mask: [N, L]

        # SSIM and MS-SSIM functions require input to be in range [0, 1]
        target, pred = self.scale_01(target), self.scale_01(pred)

        # SSIM and MS-SSIM functions require input [N, C, H, W]
        target = self.unpatchify(target, patch_embed_psize, input_channels)
        pred = self.unpatchify(pred, patch_embed_psize, input_channels)

        # By default perform SSIM on reconstructed masked patches only (when used stand-alone)
        # Optionally, if mask is None, perform SSIM on all patches (reconstructed and visible)
        if mask is not None:
            if patch_embed_psize is None or input_channels is None:
                raise ValueError(
                    "patch_embed_psize and input_channels must be provided if mask is provided"
                )
            mask = mask.unsqueeze(-1).repeat(
                1, 1, patch_embed_psize**2 * 3
            )  # (N, H*W, p*p*3)
            mask = self.unpatchify(
                mask, p=patch_embed_psize, c=input_channels
            )  # 1 is removing, 0 is keeping

            target = target * mask
            pred = pred * mask

        return 1 - ms_ssim(pred, target, data_range=1, size_average=True)

    def forward_loss_mse_ssim(self, target, pred, mask=None, weight=0.1, **kwargs):
        # combines mse and ssim loss
        # Loss on only reconstructed masked patches
        loss1 = self.forward_loss_mse(target, pred, mask=mask, **kwargs)
        # SSIM on whole reconstruction (not just reconstructed masked patches)
        # - to enforce global structural consistency
        loss2 = self.forward_loss_ssim(target, pred, mask=mask, **kwargs)
        # sum of the two losses
        return loss1 + weight * loss2

    def forward_loss_mse_ms_ssim(self, target, pred, mask=None, weight=0.1, **kwargs):
        # combines mse and ms-ssim loss
        # Loss on only reconstructed masked patches
        loss1 = self.forward_loss_mse(target, pred, mask=mask, **kwargs)
        # SSIM on whole reconstruction (not just reconstructed masked patches)
        # - to enforce global structural consistency
        loss2 = self.forward_loss_ms_ssim(target, pred, mask=mask, **kwargs)
        # sum of the two losses
        return loss1 + weight * loss2

    def forward_loss(
        self,
        target,
        pred,
        mask=None,
        patch_embed_psize=None,
        input_channels=None
    ):
        # patch_embed_psize and input_channels are required if passing in a full images as target
        #       Imgs: [N, C, H, W]   <= patchified by process_target below
        # Pred: [N, L, p*p*3]
        # Mask: [N, L]
        if patch_embed_psize is not None and input_channels is not None:
            target = self.process_target(target, patch_embed_psize, input_channels)

        return self.__forward_loss(
            target,
            pred,
            mask=mask,
            patch_embed_psize=patch_embed_psize,
            input_channels=input_channels
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}
