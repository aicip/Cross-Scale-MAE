from .MAE_ViT_MsLd import MAE_ViT_MsLd


class MAE_ViT_MsLdLe(MAE_ViT_MsLd):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        loss_e=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If None, use the same loss as for the reconstruction (decoder projection head)
        self.loss_e = loss_e.lower() if loss_e is not None else self.loss
        # get the loss function from class based on string
        self.__forward_loss_e = getattr(self, f"forward_loss_{self.loss_e}")
        print(f"__forward_loss_e: {self.loss_e} -> {self.__forward_loss_e.__name__}")

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        mask_seed: int = None,
        return_embeds=False,
        consistent_mask=False,
        targets=None
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

        # Latent loss between original and crop
        loss_e = self.__forward_loss_e(enc_emb_orig, enc_emb_crop)

        # Reconstruction loss + latent loss
        loss_de = loss_d + loss_e

        if not return_embeds:
            return loss_de, pred_orig, mask_orig

        return (
            loss_de,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )
