from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block, PatchEmbed
from xformers.factory import xFormer, xFormerConfig
from models_mae.MAE_ViT_Shared import MAE_ViT_Shared

from util.pos_embed import get_2d_sincos_pos_embed


class MAE_ViT_Baseline(MAE_ViT_Shared):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        input_size=128,
        input_channels=3,
        patch_size=16,  # Must be multiple of input_size
        mask_ratio=0.75,
        dim_model=1024,
        # Encoder parameters
        encoder_num_layers=24,
        encoder_num_heads=16,  # Must be multiple of dim_model
        # Decoder parameters
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,  # Must be multiple of decoder_embed_dim
        # Residual parameters
        residual_norm_style="post",
        residual_dropout=0.0,
        # Feedforward parameters
        ffn_name="MLP",  # Note: If use_xformers=False, only MLP is supported
        ffn_activation="gelu",  # Note: if use_xformers=False, only gelu is supported
        ffn_ratio=4,
        ffn_dropout=0.0,
        # Attention parameters
        attn_name="scaled_dot_product",
        attn_dropout=0.0,
        # Other parameters
        norm_layer=partial(
            nn.LayerNorm, eps=1e-6
        ),  # Note: Only used if use_xformers=False
        use_xformers=False,
        device=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.input_channels = input_channels
        self.patch_size = int(patch_size)
        self.dim_model = dim_model
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = mask_ratio
        self.use_xformers = use_xformers
        self.device = device

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        assert input_size % self.patch_size == 0

        if not use_xformers:
            assert (
                attn_name == "scaled_dot_product"
            ), f"Attention {attn_name} not supported with use_xformers=False, as Timm's implementation uses scaled_dot_product"
            assert (
                ffn_name == "MLP"
            ), f"Feedforward {ffn_name} not supported with use_xformers=False, as Timm's implementation uses MLP"
            assert (
                ffn_activation == "gelu"
            ), f"Feedforward activation {ffn_activation} not supported with use_xformers=False, as Timm's implementation uses gelu"

        self.patch_embed = PatchEmbed(
            input_size, self.patch_size, input_channels, dim_model
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, dim_model), requires_grad=False
        )  # fixed sin-cos embedding

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(dim_model, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        if use_xformers:
            print("Using xformers")
            encoder_config = xFormerConfig(
                [
                    {
                        "reversible": False,  # This decreases memory usage but increases latency
                        "block_type": "encoder",
                        "num_layers": encoder_num_layers,
                        "dim_model": dim_model,
                        "residual_norm_style": residual_norm_style,
                        "multi_head_config": {
                            "num_heads": encoder_num_heads,
                            "residual_dropout": residual_dropout,
                            "attention": {
                                "name": attn_name,
                                "dropout": attn_dropout,
                                "seq_len": self.num_patches
                                + 1,  # This adds the mask token
                                "causal": False,
                                "use_rotary_embeddings": False,  # TODO: Check if this would be useful
                            },
                        },
                        "feedforward_config": {
                            "name": ffn_name,
                            "dropout": ffn_dropout,
                            "activation": ffn_activation,
                            "hidden_layer_multiplier": ffn_ratio,
                        },
                    }
                ]
            )
            self.encoder = xFormer.from_config(encoder_config)

            decoder_config = xFormerConfig(
                [
                    {
                        "reversible": False,
                        # Using encoder here since the rest of the decoder parts are handled manually (see below)
                        "block_type": "encoder",
                        "num_layers": decoder_num_layers,
                        "dim_model": decoder_embed_dim,
                        "residual_norm_style": residual_norm_style,
                        "multi_head_config": {
                            "num_heads": decoder_num_heads,
                            "residual_dropout": residual_dropout,
                            "attention": {
                                "name": attn_name,
                                "dropout": attn_dropout,
                                "seq_len": self.num_patches
                                + 1,  # This adds the mask token
                                "causal": False,
                                "use_rotary_embeddings": False,  # TODO: Check if this would be useful
                            },
                        },
                        "feedforward_config": {
                            "name": ffn_name,
                            "dropout": ffn_dropout,
                            "activation": ffn_activation,
                            "hidden_layer_multiplier": ffn_ratio,
                        },
                    }
                ]
            )
            self.decoder = xFormer.from_config(decoder_config)
        else:
            print("Using Timm")
            encoder_blocks = [
                Block(
                    dim=dim_model,
                    num_heads=encoder_num_heads,
                    mlp_ratio=ffn_ratio,
                    qkv_bias=True,
                    drop=ffn_dropout,
                    attn_drop=attn_dropout,
                    norm_layer=norm_layer,
                    drop_path=residual_dropout,
                )
                for _ in range(encoder_num_layers)
            ]
            self.encoder = nn.ModuleList(encoder_blocks)

            decoder_blocks = [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=ffn_ratio,
                    qkv_bias=True,
                    drop=ffn_dropout,
                    attn_drop=attn_dropout,
                    norm_layer=norm_layer,
                    drop_path=residual_dropout,
                )
                for _ in range(decoder_num_layers)
            ]
            self.decoder = nn.ModuleList(decoder_blocks)

        # decoder to patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, self.patch_size**2 * input_channels, bias=True
        )
        # --------------------------------------------------------------------------
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.encoder_norm = norm_layer(dim_model)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        encoder_pos_embed = get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.encoder_pos_embed.data.copy_(
            torch.from_numpy(encoder_pos_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask_ratio):
        # TODO: Test out adding random noise to the input
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.encoder_pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        if self.use_xformers:
            x = self.encoder(x)
        else:
            for blk in self.encoder:
                x = blk(x)
        # LayerNorm
        self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x_embed = x + self.decoder_pos_embed

        # apply Transformer blocks
        if self.use_xformers:
            x_embed = self.decoder(x_embed)
        else:
            for blk in self.decoder:
                x_embed = blk(x_embed)
        # LayerNorm
        x_embed = self.decoder_norm(x_embed)

        # predictor projection & remove cls token
        x_pred = self.decoder_pred(x_embed)[:, 1:, :]

        return x_pred, x_embed

    def forward(self, imgs, mask_ratio=0.75,
                mask_seed=None, return_embeds=False):
        if mask_seed is not None:
            torch.manual_seed(mask_seed)

        encoder_embed, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        decoder_pred, decoder_embed = self.forward_decoder(
            encoder_embed, ids_restore
        )  # [N, L, p*p*3]

        loss = self.forward_loss(
            imgs,
            decoder_pred,
            mask,
            self.patch_embed.patch_size[0],
            self.input_channels
        )

        if not return_embeds:
            return loss, decoder_pred, mask
        else:
            return loss, decoder_pred, mask, encoder_embed, decoder_embed
