from .MLP import MLP

from .MAE_ViT_MsLd import MAE_ViT_MsLd


class MAE_ViT_MsLdCd(MAE_ViT_MsLd):
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
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False
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

        # Cross decoder loss between original and crop
        cross_pred = self.predictor(dec_emb_crop[:, 1:, :])
        cross_target = dec_emb_orig[:, 1:, :]
        loss_cd = self.__forward_loss_cd(cross_target, cross_pred)

        # Reconstruction loss + cross decoder loss
        loss_d_cd = loss_d + loss_cd

        if not return_embeds:
            return loss_d_cd, pred_orig, mask_orig

        return (
            loss_d_cd,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )
