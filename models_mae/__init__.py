# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ShuntedTransformer: https://github.com/OliverRensu/Shunted-Transformer
# --------------------------------------------------------

from models_mae.MAE_ViT_MsLd import MAE_ViT_MsLd
from models_mae.MAE_ViT_MsLdCd import MAE_ViT_MsLdCd
from models_mae.MAE_ViT_MsLdCe import MAE_ViT_MsLdCe
from models_mae.MAE_ViT_MsLdLe import MAE_ViT_MsLdLe
from models_mae.MAE_ViT_MsLdCeCd import MAE_ViT_MsLdCeCd
from models_mae.MAE_ViT_MsLdLeCd import MAE_ViT_MsLdLeCd
from .MAE_ViT_Baseline import *
from .models_mae_cross import *
from .models_mae_crossv2 import *
from .models_mae_shunted import *
from .models_mae_shunted_cross import *


# NOTE: Actual ViT Tiny is: embed_dim=192, depth=12, num_heads=3
args_mae_vit_tiny = {
    "dim_model": 128,
    "encoder_num_layers": 4,
    "encoder_num_heads": 8,
    "decoder_embed_dim": 256,
    "decoder_num_layers": 4,
    "decoder_num_heads": 8,
}

# NOTE: Actual ViT Small is: embed_dim=384, depth=12, num_heads=6
args_mae_vit_small = {
    "dim_model": 512,
    "encoder_num_layers": 8,
    "encoder_num_heads": 8,
    "decoder_embed_dim": 512,
    "decoder_num_layers": 8,
    "decoder_num_heads": 16,
}

args_mae_vit_base = {
    "dim_model": 768,
    "encoder_num_layers": 12,
    "encoder_num_heads": 12,
    "decoder_embed_dim": 512,
    "decoder_num_layers": 8,
    "decoder_num_heads": 16,
}

args_mae_vit_large = {
    "dim_model": 1024,
    "encoder_num_layers": 24,
    "encoder_num_heads": 16,
    "decoder_embed_dim": 512,
    "decoder_num_layers": 8,
    "decoder_num_heads": 16,
}

args_mae_vit_huge = {
    "dim_model": 1280,
    "encoder_num_layers": 32,
    "encoder_num_heads": 16,
    "decoder_embed_dim": 512,
    "decoder_num_layers": 8,
    "decoder_num_heads": 16,
}


# --- MAE Models --- #
def mae_vit_base(**kwargs):
    model = MAE_ViT_Baseline(
        **args_mae_vit_base,
        **kwargs,
    )
    return model


def mae_vit_base_MsLd(**kwargs):
    model = MAE_ViT_MsLd(
        **args_mae_vit_base,
        **kwargs,
    )
    return model


def mae_vit_base_MsLdLe(**kwargs):
    model = MAE_ViT_MsLdLe(
        **args_mae_vit_base,
        **kwargs,
    )
    return model


def mae_vit_base_MsLdCd(**kwargs):
    model = MAE_ViT_MsLdCd(
        **args_mae_vit_base,
        **kwargs,
    )
    return model


def mae_vit_base_MsLdCe(**kwargs):
    model = MAE_ViT_MsLdCe(
        **args_mae_vit_base,
        **kwargs,
    )
    return model


def mae_vit_base_MsLdLeCd(**kwargs):
    model = MAE_ViT_MsLdLeCd(
        **args_mae_vit_base,
        **kwargs,
    )
    return model


def mae_vit_base_MsLdCeCd(**kwargs):
    model = MAE_ViT_MsLdCeCd(
        **args_mae_vit_base,
        **kwargs,
    )
    return model


def mae_vit_base_cross(**kwargs):
    model = MAE_ViT_OldCross(
        **args_mae_vit_base,
        # Cross-V1 specific
        predictor_hidden_size=2048,
        **kwargs,
    )
    return model


def mae_vit_base_crossV2(**kwargs):
    model = MAE_ViT_OldCrossV2(
        **args_mae_vit_base,
        # Cross-V2 specific
        loss_latent_weight=1.0,
        losses_pred_reduction="sum",
        lossed_latent_reduction="mean",
        **kwargs,
    )
    return model


def mae_vit_large(**kwargs):
    model = MAE_ViT_Baseline(
        dim_model=1024,
        **kwargs,
    )
    return model


def mae_vit_huge(**kwargs):
    model = MAE_ViT_Baseline(
        **args_mae_vit_huge,
        **kwargs,
    )
    return model


# --- Shunted Models --- #
def mae_vit_tiny_shunted_2st(**kwargs):
    model = MaskedAutoencoderShuntedViT(
        # Encoder
        dim_model=[64, 128],
        encoder_num_layers=[2, 4],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_tiny = mae_vit_tiny_shunted_2st


def mae_vit_tiny_shunted_2st_cross(**kwargs):
    model = MaskedAutoencoderShuntedViTCross(
        # Encoder
        dim_model=[64, 128],
        encoder_num_layers=[2, 4],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_tiny_cross = mae_vit_tiny_shunted_2st_cross


def mae_vit_mini_shunted_2st(**kwargs):
    model = MaskedAutoencoderShuntedViT(
        # Encoder
        dim_model=[128, 256],
        encoder_num_layers=[2, 4],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[4, 2],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_mini = mae_vit_mini_shunted_2st


def mae_vit_small_shunted_2st(**kwargs):
    model = MaskedAutoencoderShuntedViT(
        # Encoder
        dim_model=[256, 512],
        encoder_num_layers=[4, 8],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_small = mae_vit_small_shunted_2st


def mae_vit_small_shunted_2st_cross(**kwargs):
    model = MaskedAutoencoderShuntedViTCross(
        # Encoder
        dim_model=[256, 512],
        encoder_num_layers=[4, 8],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_small_cross = mae_vit_small_shunted_2st_cross


def mae_vit_base_shunted_2st(**kwargs):
    model = MaskedAutoencoderShuntedViT(
        # Encoder
        dim_model=[512, 768],
        encoder_num_layers=[8, 12],
        encoder_num_heads=[8, 12],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_base = mae_vit_base_shunted_2st


def mae_vit_base_shunted_2st_cross(**kwargs):
    model = MaskedAutoencoderShuntedViTCross(
        # Encoder
        dim_model=[512, 768],
        encoder_num_layers=[8, 12],
        encoder_num_heads=[8, 12],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_base_cross = mae_vit_base_shunted_2st_cross