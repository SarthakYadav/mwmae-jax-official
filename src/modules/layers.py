import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Union, Callable, Optional
from .droppath import DropPath
from .mlp import Mlp
from .model_utils import constant_init
from itertools import repeat
import collections.abc


dense_kernel_init = nn.initializers.xavier_uniform()
position_table_init = nn.initializers.variance_scaling(0.02, "fan_in", "truncated_normal")


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


class Attention(nn.Module):
    """
    Default multihead attention
    """
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(self.dtype)
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5
        qkv_layer = nn.Dense(self.dim * 3, dtype=self.dtype, use_bias=self.qkv_bias, kernel_init=dense_kernel_init)
        proj_layer = nn.Dense(self.dim, dtype=self.dtype, kernel_init=dense_kernel_init)

        B, N, C = x.shape
        qkv = qkv_layer(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv
        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = nn.softmax(attn, axis=-1)
        if self.attn_drop != 0:
            attn = nn.Dropout(self.attn_drop, deterministic=not train, name="attn_drop_layer")(attn)
        x = jnp.swapaxes((attn @ v), 1, 2).reshape(B, N, C)
        x = proj_layer(x)
        if self.proj_drop != 0:
            x = nn.Dropout(self.proj_drop, deterministic=not train, name="proj_drop_layer")(x)
        x = x.astype(self.dtype)
        return x


def window_partition1d(x, window_size):
    B, W, C = x.shape
    x = x.reshape(B, W // window_size, window_size, C)
    windows = x.reshape(-1, window_size, C)
    return windows


def window_reverse1d(windows, window_size, W: int):
    B = int(windows.shape[0] / (W / window_size))
    x = windows.reshape(B, W // window_size, window_size, -1)
    x = x.reshape(B, W, -1)
    return x


def get_relative_position_index1d(win_w, dtype=jnp.float32):
    # get pair-wise relative position index for each token inside the window
    coords_flatten = jnp.stack(jnp.meshgrid(jnp.arange(win_w)))
    
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ww, Ww
    

    relative_coords = jnp.transpose(relative_coords, (1, 2, 0))

    relative_coords = relative_coords.at[:, :, 0].set(relative_coords[:, :, 0] + (win_w - 1))
    return relative_coords.sum(-1) #.astype(dtype)  # Wh*Ww, Wh*Ww


class WindowedAttentionHead(nn.Module):
    head_dim: int
    window_size: int
    shift_windows: bool = False
    attn_drop: Optional[float] = 0.
    position_bias_table_init: Optional[Callable] = position_table_init
    dtype: Any = jnp.float32

    def setup(self):
        # pos_tab_init_fn = nn.initializers.variance_scaling(0.02, "fan_in", "truncated_normal", dtype=self.dtype)
        self.relative_position_bias_table = self.param(
            "relative_position_bias_table",
            position_table_init,                    # confirm that initialization is identical to torch version
            (2 * self.window_size - 1, 1))
        self.relative_position_index = self.variable(
            "buffers", "relative_position_index",
            init_fn=partial(
                get_relative_position_index1d, win_w=self.window_size
            )
        )
        self.scale = self.head_dim ** -0.5
        self.window_area = self.window_size * 1
        if self.attn_drop != 0:
            self.drop_layer = nn.Dropout(rate=self.attn_drop, name="attn_drop_layer")
        else:
            self.drop_layer = None
        if self.shift_windows:
            self.shift_size = self.window_size // 2
        else:
            self.shift_size = 0
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

    def __call__(self, q, k, v, train: bool = True):
        B, W, C = q.shape
        q = q.astype(self.dtype)
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)

        mask = None
        cnt = 0
        if self.shift_size > 0:
            img_mask = jnp.zeros((1, W, 1))
            for w in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)):
                img_mask = img_mask.at[:, w, :].set(cnt)
                cnt += 1
            mask_windows = window_partition1d(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(-1, self.window_size)
            # mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            mask = mask_windows[:, None, Ellipsis] - mask_windows[:,:,None,Ellipsis]
            mask = jnp.where(mask != 0., -100., mask)
            mask = jnp.where(mask == 0., 0., mask)

            q = jnp.roll(q, shift=-self.shift_size,axis=1)
            k = jnp.roll(k, shift=-self.shift_size,axis=1)
            v = jnp.roll(v, shift=-self.shift_size,axis=1)

        else:
            mask = None

        q = window_partition1d(q, self.window_size)
        k = window_partition1d(k, self.window_size)
        v = window_partition1d(v, self.window_size)

        attn = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        if train:
            attn = attn + self._get_rel_pos_bias()
        else:
            e = self._copied_rel_pos_bias()
            attn = attn + e

        if mask is not None:
            B_, N, _ = attn.shape
            num_win = mask.shape[0]
            attn = attn.reshape(B_//num_win, num_win, N, N) + mask[None, Ellipsis]
            attn = attn.reshape(-1, N, N)
            attn = nn.softmax(attn, axis=-1)
        else:
            attn = nn.softmax(attn, axis=-1)

        if self.drop_layer is not None:
            attn = self.drop_layer(attn, deterministic=not train)

        x = (attn @ v)

        # merge windows
        shifted_x = window_reverse1d(x, self.window_size, W=W)

        if self.shift_size > 0:
            x = jnp.roll(shifted_x, shift=self.shift_size,axis=1)
        else:
            x = shifted_x
        
        x = x.astype(self.dtype)
        
        return x, attn
    
    def _get_rel_pos_bias(self):
        relative_position_index = self.relative_position_index.value
        relative_position_bias = self.relative_position_bias_table[
            jnp.array(relative_position_index.reshape(-1))].reshape(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias

    def _copied_rel_pos_bias(self):
        # from jax.experimental import io_callback
        relative_position_index = self.relative_position_index.value.copy()

        tmp = self.relative_position_bias_table.copy()
        relative_position_bias = tmp[relative_position_index.reshape(-1)].reshape(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias


class AttentionHead(nn.Module):
    head_dim: int
    attn_drop: Optional[float] = 0.
    dtype: Any = jnp.float32

    def setup(self):
        self.scale = self.head_dim ** -0.5
        if self.attn_drop != 0:
            self.drop_layer = nn.Dropout(rate=self.attn_drop, name="attn_drop_layer")
        else:
            self.drop_layer = None
    
    def __call__(self, q, k, v, train: bool = True):
        B, W, C = q.shape
        q = q.astype(self.dtype)
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)
        attn = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        attn = nn.softmax(attn, axis=-1)

        if self.drop_layer is not None:
            attn = self.drop_layer(attn, deterministic=not train)
        
        x = (attn @ v)
        x = x.astype(self.dtype)
        # returning attn for ease of use
        return x, attn


class WindowedMultiHeadAttention(nn.Module):
    dim: int
    window_sizes: Union[list, tuple, int]
    shift_windows: bool = False
    num_heads: int = 8
    qkv_bias: bool = False
    attn_drop: Optional[float] = 0.0
    proj_drop: Optional[float] = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        self.head_dim = self.dim // self.num_heads

        self.qkv = nn.Dense(self.dim * 3, dtype=self.dtype, use_bias=self.qkv_bias, kernel_init=dense_kernel_init)
        if type(self.window_sizes) == int:
            window_sizes = _ntuple(self.num_heads)(self.window_sizes)
        else:
            window_sizes = self.window_sizes
            assert len(window_sizes) == self.num_heads
        
        attn_heads = []
        for i in range(self.num_heads):
            ws_i = window_sizes[i]
            if ws_i == 0:
                attn_heads.append(AttentionHead(self.head_dim, self.attn_drop, dtype=self.dtype))
            else:
                attn_heads.append(WindowedAttentionHead(
                    self.head_dim,
                    window_size=ws_i,
                    shift_windows=self.shift_windows,
                    attn_drop=self.attn_drop,
                    dtype=self.dtype
                ))
        self.attn_heads = attn_heads
        self.proj = nn.Dense(self.dim, dtype=self.dtype, kernel_init=dense_kernel_init)
        if self.proj_drop != 0:
            self.drop_layer = nn.Dropout(rate=self.attn_drop, name="proj_drop_layer")
        else:
            self.drop_layer = None
    
    def __call__(self, x, train: bool = True):
        B, N, C = x.shape
        x = x.astype(self.dtype)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose((2, 3, 0, 1, 4))
        q, k, v = qkv
        o = []
        for i in range(self.num_heads):
            head_i, attn_i = self.attn_heads[i](q[i], k[i], v[i], train=train)
            head_i = head_i[jnp.newaxis, Ellipsis]
            o.append(head_i)
        o = jnp.concatenate(o, axis=0)
        o = jnp.transpose(o, (1, 2, 0, 3)).reshape(B, N, -1)
        o = self.proj(o)
        if self.drop_layer is not None:
            o = self.drop_layer(o, deterministic=not train)
        o = o.astype(self.dtype)
        return o


class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x):
        gamma = self.param("gamma", partial(constant_init, constant=self.init_values), [self.dim])
        return x * gamma


class BNWrapper(nn.Module):
    use_running_average: bool = True
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.BatchNorm(use_running_average=not train, use_bias=self.use_bias,
                             use_scale=self.use_scale, name='head_norm')(x)
        return x


class Block(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    drop: float = 0.
    attn_drop: float = 0.
    init_values: Any = None
    drop_path: float = 0.
    act_layer: Union[Callable, nn.Module] = nn.gelu
    norm_layer: Union[Callable, nn.Module] = nn.LayerNorm
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        outputs1 = self.norm_layer(dtype=self.dtype)(x)
        outputs1 = Attention(self.dim, num_heads=self.num_heads,
                             qkv_bias=self.qkv_bias, attn_drop=self.attn_drop,
                             dtype=self.dtype,
                             proj_drop=self.drop)(outputs1, train=train)

        if self.init_values is not None:
            outputs1 = LayerScale(self.dim, init_values=self.init_values)(outputs1)

        if self.drop_path > 0.:
            outputs1 = DropPath(self.drop_path)(outputs1, train=train)

        x = x + outputs1

        outputs2 = self.norm_layer(dtype=self.dtype)(x)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        outputs2 = Mlp(hidden_features=mlp_hidden_dim, out_features=self.dim,
                       drop=self.drop, dtype=self.dtype, kernel_init=dense_kernel_init,
                       activation=self.act_layer)(outputs2, train=train)

        if self.init_values is not None:
            outputs2 = LayerScale(self.dim, init_values=self.init_values)(outputs2)

        if self.drop_path > 0.:
            outputs2 = DropPath(self.drop_path)(outputs2, train=train)
        x = x + outputs2
        x = x.astype(self.dtype)
        return x


class MWMHABlock(nn.Module):
    dim: int
    num_heads: int
    window_sizes: Union[list, tuple, int]
    shift_windows: bool = False
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    drop: float = 0.
    attn_drop: float = 0.
    init_values: Any = None
    drop_path: float = 0.
    act_layer: Union[Callable, nn.Module] = nn.gelu
    norm_layer: Union[Callable, nn.Module] = nn.LayerNorm
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        outputs1 = self.norm_layer(dtype=self.dtype)(x)
        outputs1 = WindowedMultiHeadAttention(
            self.dim,
            window_sizes=self.window_sizes,
            shift_windows=self.shift_windows,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            dtype=self.dtype,
            name='wmha_block'
        )(outputs1, train=train)

        if self.init_values is not None:
            outputs1 = LayerScale(self.dim, init_values=self.init_values)(outputs1)

        if self.drop_path > 0.:
            outputs1 = DropPath(self.drop_path)(outputs1, train=train)

        x = x + outputs1

        outputs2 = self.norm_layer(dtype=self.dtype)(x)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        outputs2 = Mlp(hidden_features=mlp_hidden_dim, out_features=self.dim,
                       drop=self.drop, dtype=self.dtype, kernel_init=dense_kernel_init,
                       activation=self.act_layer)(outputs2, train=train)

        if self.init_values is not None:
            outputs2 = LayerScale(self.dim, init_values=self.init_values)(outputs2)

        if self.drop_path > 0.:
            outputs2 = DropPath(self.drop_path)(outputs2, train=train)
        x = x + outputs2
        x = x.astype(self.dtype)
        return x
