import jax
from functools import partial
import jax.numpy as jnp
import flax.linen as nn
from .patch_embed import PatchEmbed
from .layers import Block, MWMHABlock
from .pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from .model_utils import batched_gather
from ..helpers.utilities import TrainingMode
from typing import Optional, Callable, Any, List, Union
from einops import rearrange


__all__ = [
    "MAE",
    "MWMAE",
    "mae_tiny",
    "mae_small",
    "mae_medium",
    "mae_base",
    "mae_large",
    "mae_huge"
]


def random_masking(x, mask_ratio, rng, dtype=jnp.float32):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (100 - mask_ratio*100)/100)      # to fix (1-0.8) == 0.19999999!

    noise = jax.random.uniform(rng, (N, L), dtype=dtype)

    # sort noise for each sample
    ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
    ids_restore = jnp.argsort(ids_shuffle, axis=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = batched_gather(x, ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = jnp.ones((N, L), dtype=dtype)
    mask = mask.at[:, :len_keep].set(0)

    mask = batched_gather(mask, ids_restore)
    return x_masked, mask, ids_restore


class MAEBase(nn.Module):
    img_size: List[int] = (200, 80)
    patch_size: List[int] = (4, 16)
    in_chans: int = 1
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 384
    decoder_depth: int = 4
    decoder_num_heads: int = 8
    mlp_ratio: int = 4
    mask_ratio: float = 0.8
    norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm
    use_cls_token: bool = False
    mode: TrainingMode = TrainingMode.MAE
    supervised_globalpool: bool = False
    supervised_num_classes: int = None
    dtype: Any = jnp.float32

    def setup(self):
        assert self.mode in [TrainingMode.MAE, TrainingMode.MULTICLASS, TrainingMode.MULTILABEL]
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.embed_dim, dtype=self.dtype)
        self.num_patches = self.patch_embed.num_patches
        total_patches = self.num_patches
        if self.use_cls_token:
            total_patches += 1
            self.cls_token = self.param("cls_token", nn.initializers.normal(0.02, dtype=self.dtype),
                                   [1, 1, self.embed_dim])
        else:
            self.cls_token = None
        self.total_patches = total_patches
        self.pos_embed = self.variable("params", "pos_embed",
                                    init_fn=partial(get_2d_sincos_pos_embed,
                                                    embed_dim=self.embed_dim,
                                                    grid_size=self.patch_embed.grid_size,
                                                    cls_token=self.use_cls_token, expand_first_dim=True,
                                                    dtype=self.dtype),
                                    )
        
        self.blocks = None
        self.encoder_norm = self.norm_layer(dtype=self.dtype, name="encoder_norm")
        self.decoder_embed = nn.Dense(self.decoder_embed_dim, dtype=self.dtype, use_bias=True)
        self.mask_token = self.param("mask_token", nn.initializers.normal(0.02, dtype=self.dtype),
                                     [1, 1, self.decoder_embed_dim])

        self.decoder_pos_embed = self.variable("params", "decoder_pos_embed",
                                               init_fn=partial(get_1d_sincos_pos_embed,
                                                               embed_dim=self.decoder_embed_dim,
                                                               grid_size=self.patch_embed.num_patches,
                                                               cls_token=self.use_cls_token, expand_first_dim=True,
                                                               dtype=self.dtype))
        self.decoder_blocks = None
        self.decoder_pred = nn.Dense(self.img_patch_dim(), dtype=self.dtype, use_bias=True)
        self.decoder_norm = self.norm_layer(dtype=self.dtype, name="decoder_norm")

        # supervised section
        if self.mode != TrainingMode.MAE:
            if self.supervised_globalpool:
                self.fc_norm = self.norm_layer(dtype=self.dtype, name="fc_norm")
            self.head = nn.Dense(self.supervised_num_classes, dtype=self.dtype, name="head")
    
    def img_patch_dim(self):
        patch_size = self.patch_embed.patch_size
        return patch_size[0] * patch_size[1] * self.in_chans
    
    def get_patch_size(self):
        return self.patch_embed.patch_size
    
    def get_patch_grid(self):
        return self.patch_embed.grid_size
    
    def patchify(self, imgs):
        """
        imgs: (N, H, W, 1)
        x: (N, L, patch_size[0]*patch_size[0]*in_chans)
        """
        ph, pw = self.patch_embed.patch_size
        h, w = self.patch_embed.grid_size
        
        x = imgs.reshape((imgs.shape[0], h, ph, w, pw, self.in_chans))
        x = jnp.einsum('nhpwqc->nhwpqc', x)
        x = x.reshape((imgs.shape[0], h * w, self.img_patch_dim()))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, H, W, 3)
        """
        ph, pw = self.patch_embed.patch_size
        h, w = self.patch_embed.grid_size
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, ph, pw, self.in_chans))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * ph, w * pw, self.in_chans))
        return imgs

    def forward_encoder(self, x, mask_ratio, train: bool = True, rng=None):
        x = self.patch_embed(x)
        pos_embed = self.pos_embed.value.astype(jnp.float32)
        if self.use_cls_token:
            x = x + pos_embed[:, 1:, :]
        else:
            x = x + pos_embed[:, :, :]
        x, mask, ids_restore = random_masking(x, mask_ratio, rng, self.dtype)
        if self.use_cls_token:
            cls_token = self.cls_token + pos_embed[:, :1, :]
            cls_tokens = jnp.broadcast_to(cls_token, (x.shape[:1] + cls_token.shape[1:]))
            x = jnp.concatenate([cls_tokens, x], axis=1)
        for blk in self.blocks:
            x = blk(x, train=train)
        x = self.encoder_norm(x)
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore, train: bool = True):
        x = self.decoder_embed(x)
        mask_tokens = jnp.broadcast_to(self.mask_token,
                                       (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], x.shape[-1]))
        if self.use_cls_token:
            x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)
            x_ = batched_gather(x_, ids_restore)
            x = jnp.concatenate([x[:, :1, :], x_], axis=1)
        else:
            x = jnp.concatenate([x, mask_tokens], axis=1)
            x = batched_gather(x, ids_restore)
        decoder_pos_embed = self.decoder_pos_embed.value.astype(jnp.float32)
        x = x + decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x, train=train) 
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        if self.use_cls_token:
            x = x[:, 1:, :] 
        return x

    def forward_features(self, x, train: bool = False, rng=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        pos_embed = self.pos_embed.value.astype(jnp.float32)
        if self.use_cls_token:
            x = x + pos_embed[:, 1:, :]
            cls_token = self.cls_token + pos_embed[:, :1, :]
            cls_tokens = jnp.broadcast_to(cls_token, (x.shape[:1] + cls_token.shape[1:]))
            x = jnp.concatenate([cls_tokens, x], axis=1)
        else:
            x = x + pos_embed[:, :, :]
        for blk in self.blocks:
            x = blk(x, train=train)
        if self.mode == TrainingMode.MAE:
            x = self.encoder_norm(x)
            outcome = x[:, 1:, :] if self.use_cls_token else x
            grid_size = self.patch_embed.grid_size
            t, f = grid_size
            outcome = rearrange(outcome, 'b (t f) d -> b t (f d)', f=f, d=self.embed_dim)
        else:
            if self.supervised_globalpool:
                outcome = x[:, 1:, :].mean(axis=1) if self.use_cls_token else x.mean(axis=1)
                outcome = self.fc_norm(outcome)
            else:
                outcome = self.encoder_norm(x)[:, 0]
            outcome = self.head(outcome)
        outcome = outcome.astype(self.dtype)
        return outcome

    def forward_mae(self, imgs, train: bool = True, rng = None):
        imgs = imgs.astype(self.dtype)
        if rng is None:
            rng = self.make_rng("random_masking")
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio, train=train, rng=rng)
        pred = self.forward_decoder(latent, ids_restore, train=train)
        target = self.patchify(jax.lax.stop_gradient(imgs))
        pred = pred.astype(self.dtype)
        target = target.astype(self.dtype)
        return pred, target, mask

    def __call__(self, imgs, train: bool = True, rng=None):
        if self.mode != TrainingMode.MAE:
            return self.forward_features(imgs, train=train, rng=rng)
        else:
            return self.forward_mae(imgs, train=train, rng=rng)


class MAE(MAEBase):
    use_cls_token: bool = True
    def setup(self):
        super().setup()

        self.blocks = [
            Block(
                self.embed_dim, 
                self.num_heads,
                self.mlp_ratio,
                qkv_bias=True,
                norm_layer=self.norm_layer,
                dtype=self.dtype,
                name="encoder_block_{:02d}".format(i)) for i in range(self.depth)
            ]
        self.decoder_blocks = [
            Block(
                self.decoder_embed_dim,
                self.decoder_num_heads,
                self.mlp_ratio,
                qkv_bias=True,
                norm_layer=self.norm_layer,
                dtype=self.dtype,
                name="decoder_block_{:02d}".format(i)) for i in range(self.decoder_depth)]


class MWMAE(MAEBase):
    window_sizes: Union[list, tuple, int] = 0
    decoder_window_sizes: Union[list, tuple, int] = 0
    shifting_windows: bool = False

    def setup(self):
        super().setup()
    
        if self.window_sizes == 0:
            print("Using optimized MHA Block for Encoder because no windows are being used")
            self.blocks = [
                Block(
                    self.embed_dim,
                    self.num_heads,
                    self.mlp_ratio,
                    qkv_bias=True, 
                    norm_layer=self.norm_layer, 
                    dtype=self.dtype,
                    name="encoder_block_{:02d}".format(i)) for i in range(self.depth)]
        else:
            self.blocks = [
                MWMHABlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    window_sizes=self.window_sizes,
                    shift_windows=not (i % 2 == 0) and self.shifting_windows,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=self.norm_layer,
                    dtype=self.dtype,
                    name="encoder_block_{:02d}".format(i)) for i in range(self.depth)
                ]
        
        if self.decoder_window_sizes == 0:
            print("Using optimized MHA Block for Decoder because no windows are being used")
            self.decoder_blocks = [
                Block(
                    self.decoder_embed_dim, 
                    self.decoder_num_heads, 
                    self.mlp_ratio, 
                    qkv_bias=True,
                    norm_layer=self.norm_layer,
                    dtype=self.dtype,
                    name="decoder_block_{:02d}".format(i)) for i in range(self.decoder_depth)]
        else:
            self.decoder_blocks = [
                MWMHABlock(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    window_sizes=self.decoder_window_sizes,
                    shift_windows=not (i % 2 == 0) and self.shifting_windows,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=self.norm_layer,
                    dtype=self.dtype,
                    name="decoder_block_{:02d}".format(i)) for i in range(self.decoder_depth)
                ]


encoder_configs = {
    "tiny": {
        "depth": 12, "num_heads": 3, "embed_dim": 192
    },
    "small": {
        "depth": 12, "num_heads": 6, "embed_dim": 384
    },
    "medium": {
        "depth": 12, "num_heads": 8, "embed_dim": 512
    },
    "base": {
        "depth": 12, "num_heads": 12, "embed_dim": 768
    },
    "large": {
        "depth": 24, "num_heads": 16, "embed_dim": 1024
    },
    "huge": {
        "depth": 32, "num_heads": 16, "embed_dim": 1280
    }
}


def _get_mae(encoder_name, **kwargs):
    in_chans = kwargs.pop("in_chans", 1)
    img_size = kwargs.pop("img_size", (200, 80))
    patch_size = kwargs.pop("patch_size", (4, 16))
    decoder_num_heads = kwargs.pop("decoder_num_heads", 8)
    decoder_depth = kwargs.pop("decoder_depth", 4)
    decoder_embed_dim = kwargs.pop("decoder_embed_dim", 384)
    mode = TrainingMode(kwargs.pop("mode", "mae"))

    #MWMAE params
    window_sizes = kwargs.pop("window_sizes", 0)
    shifting_windows = kwargs.pop("shifting_windows", False)
    decoder_window_sizes = kwargs.pop("decoder_window_sizes", 0)

    enc_params = encoder_configs[encoder_name]
    if window_sizes != 0 or decoder_window_sizes != 0:
        model = MWMAE(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=enc_params["embed_dim"],
            depth=enc_params["depth"],
            num_heads=enc_params["num_heads"],
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            window_sizes=window_sizes,
            decoder_window_sizes=decoder_window_sizes,
            shifting_windows=shifting_windows,
            mode=mode,
            **kwargs
        )
    else:
        model = MAE(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=enc_params["embed_dim"],
            depth=enc_params["depth"],
            num_heads=enc_params["num_heads"],
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mode=mode,
            **kwargs
        )
    return model


def mae_tiny(**kwargs):
    return _get_mae("tiny", **kwargs)


def mae_small(**kwargs):
    return _get_mae("small", **kwargs)


def mae_medium(**kwargs):
    return _get_mae("medium", **kwargs)


def mae_base(**kwargs):
    return _get_mae("base", **kwargs)


def mae_large(**kwargs):
    return _get_mae("large", **kwargs)


def mae_huge(**kwargs):
    return _get_mae("huge", **kwargs)
