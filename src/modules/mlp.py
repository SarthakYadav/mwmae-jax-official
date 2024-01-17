"""
Implementation of the MLP layer as used in recent literature for vision transformers,etc

Inspired by the PyTorch implementation in timm (https://github.com/rwightman/pytorch-image-models)
by Ross Wightman, 2022

Modified for audax by / Copyright 2022, Sarthak Yadav
"""
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable


default_init = nn.initializers.lecun_normal


class Mlp(nn.Module):
    """
    Mlp as used in Vision transformers, MLP mixers and such
    """
    hidden_features: int
    out_features: int = None
    drop: float = 0.
    activation: Callable = nn.gelu
    dtype: Any = jnp.float32
    kernel_init: Callable = default_init

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        out_features = self.out_features or self.hidden_features
        hidden_feature = self.hidden_features

        outputs = inputs.astype(self.dtype)
        outputs = nn.Dense(hidden_feature, kernel_init=self.kernel_init, dtype=self.dtype)(outputs)
        outputs = self.activation(outputs)
        if self.drop != 0:
            outputs = nn.Dropout(self.drop)(outputs, deterministic=not train)
        outputs = nn.Dense(out_features, kernel_init=self.kernel_init,  dtype=self.dtype)(outputs)
        if self.drop != 0:
            outputs = nn.Dropout(self.drop)(outputs, deterministic=not train)
        outputs = outputs.astype(self.dtype)
        return outputs
