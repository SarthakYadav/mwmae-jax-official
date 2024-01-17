import jax
import jax.numpy as jnp
import flax.linen as nn
from .model_utils import to_2tuple, to_1tuple
from typing import Optional, Callable, Union, Any
from functools import partial


class PatchEmbed(nn.Module):
    img_size: Optional[Union[tuple, int]] = 224
    patch_dim: Optional[Union[tuple, int]] = 16
    # in_chans: int = 3
    embed_dim: int = 768
    norm_layer: Optional[Callable] = None
    flatten: bool = True
    dtype: Optional[Any] = jnp.float32

    def setup(self):
        img_size = to_2tuple(self.img_size)
        patch_size = to_2tuple(self.patch_dim)
        grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.proj = nn.Conv(self.embed_dim, kernel_size=patch_size, strides=patch_size, padding='VALID',
                            kernel_init=nn.initializers.xavier_uniform(), dtype=self.dtype)

    def __call__(self, inputs, train: bool = True):
        inputs = inputs.astype(self.dtype)
        B, H, W, C = inputs.shape
        outputs = self.proj(inputs)
        if self.flatten:
            outputs = outputs.reshape(B, -1, self.embed_dim) # B,N,C shape
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)
        print("PatchEmbed outputs shape:", outputs.shape)
        outputs = outputs.astype(self.dtype)
        return outputs
