import torch
from models_mae.MAE_ViT_MsLd import MAE_ViT_MsLd, MAE_ViT_MsLd_PAIRED
from models_mae.MLP import MLP
from util.contrast_loss import NTXentLoss


class MAE_ViT_MsLdCeCd(MAE_ViT_MsLd):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        loss_cd=None,
        predictor_hidden_size=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If None, use the same loss as for the reconstruction (decoder projection head)
        self.loss_cd = loss_cd.lower() if loss_cd is not None else self.loss
        # get the loss function from class based on string
        self.__forward_loss_cd = getattr(self, f"forward_loss_{self.loss_cd}")
        print(f"__forward_loss_cd: {self.loss_cd} -> {self.__forward_loss_cd.__name__}")

        self.predictor = MLP(
            self.decoder_embed_dim, self.num_patches, predictor_hidden_size
        )

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        contr_bs=None,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False,
        **kwargs,
    ):
        (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        ) = super().forward(
            imgs,
            mask_ratio=mask_ratio,
            mask_seed=mask_seed,
            return_embeds=True,
            consistent_mask=consistent_mask,
        )

        if contr_bs:
            bs = contr_bs
        else:
            bs = imgs.shape[0]

        # Cross decoder loss between original and crop
        cross_pred = self.predictor(dec_emb_crop[:, 1:, :])
        cross_target = dec_emb_orig[:, 1:, :]
        loss_cd = self.__forward_loss_cd(cross_target, cross_pred)

        # Contrastive encoder loss between original and crop
        contrast_criterian = NTXentLoss(bs, 0.5, cos_sim=True, device=self.device)

        f1 = torch.flatten(enc_emb_orig[:, 1:, :].mean(dim=1), 1)
        f2 = torch.flatten(enc_emb_crop[:, 1:, :].mean(dim=1), 1)
        # print('f1 shape:',f1.shape)
        # print('f2 shape:',f2.shape)

        loss_ce = contrast_criterian(f1, f2)

        # Reconstruction loss + cross decoder loss
        loss_d_cd_ce = loss_d + loss_cd + loss_ce
        # loss_d_cd_ce = loss_d + loss_cd

        if not return_embeds:
            return loss_d_cd_ce, pred_orig, mask_orig

        return (
            loss_d_cd_ce,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )


class MAE_ViT_MsLdCeCd_PAIRED(MAE_ViT_MsLd_PAIRED):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        device="cuda:0",
        loss_cd=None,
        # bacth_size =128,
        predictor_hidden_size=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If None, use the same loss as for the reconstruction (decoder projection head)
        self.loss_cd = loss_cd.lower() if loss_cd is not None else self.loss
        # get the loss function from class based on string
        self.__forward_loss_cd = getattr(self, f"forward_loss_{self.loss_cd}")
        print(f"__forward_loss_cd: {self.loss_cd} -> {self.__forward_loss_cd.__name__}")

        # self.batch_size = bacth_size
        self.device = device

        self.predictor = MLP(
            self.decoder_embed_dim, self.num_patches, predictor_hidden_size
        )

        # self.contrast_criterian = NTXentLoss(self.batch_size, self.device, 0.5, cos_sim=True)

    def forward(
        self,
        imgs1,
        imgs2,
        mask_ratio=0.75,
        contr_bs=None,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False,
        **kwargs,
    ):
        (
            loss_d,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        ) = super().forward(
            imgs1,
            imgs2,
            mask_ratio=mask_ratio,
            mask_seed=mask_seed,
            return_embeds=True,
            consistent_mask=consistent_mask,
        )

        if contr_bs:
            bs = contr_bs
        else:
            bs = imgs1.shape[0]

        # Cross decoder loss between original and crop
        cross_pred = self.predictor(dec_emb_crop[:, 1:, :])
        cross_target = dec_emb_orig[:, 1:, :]
        loss_cd = self.__forward_loss_cd(cross_target, cross_pred)

        # Contrastive encoder loss between original and crop

        contrast_criterian = NTXentLoss(self.device, bs, 0.5, cos_sim=True)

        f1 = torch.flatten(enc_emb_orig[:, 1:, :].mean(dim=1), 1)
        f2 = torch.flatten(enc_emb_crop[:, 1:, :].mean(dim=1), 1)
        # print('f1 shape:',f1.shape)
        # print('f2 shape:',f2.shape)

        loss_ce = contrast_criterian(f1, f2)

        # Reconstruction loss + cross decoder loss
        loss_d_cd_ce = loss_d + loss_cd + loss_ce
        # loss_d_cd_ce = loss_d + loss_cd

        if not return_embeds:
            return loss_d_cd_ce, pred_orig, mask_orig

        return (
            loss_d_cd_ce,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )
