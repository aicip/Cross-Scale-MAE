from .MLP import MLP
from .MAE_ViT_MsLd import MAE_ViT_MsLd


class MAE_ViT_MsLdCe(MAE_ViT_MsLd):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        loss_ce=None,
        predictor_hidden_size=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If None, use the same loss as for the reconstruction (decoder projection head)
        self.loss_ce = loss_ce.lower() if loss_ce is not None else self.loss
        # get the loss function from class based on string
        self.__forward_loss_ce = getattr(self, f"forward_loss_{self.loss_ce}")
        print(f"__forward_loss_ce: {self.loss_ce} -> {self.__forward_loss_ce.__name__}")

        self.predictor = MLP(self.dim_model, self.num_patches, predictor_hidden_size)

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

        # Cross encoder loss between original and crop
        cross_pred = self.predictor(enc_emb_crop[:, 1:, :])
        cross_target = enc_emb_orig[:, 1:, :]
        loss_ce = self.__forward_loss_ce(cross_target, cross_pred)

        # Reconstruction loss + cross encoder loss
        loss_d_ce = loss_d + loss_ce

        if not return_embeds:
            return loss_d_ce, pred_orig, mask_orig

        return (
            loss_d_ce,
            pred_orig,
            mask_orig,
            (enc_emb_orig, enc_emb_crop),
            (dec_emb_orig, dec_emb_crop),
        )
