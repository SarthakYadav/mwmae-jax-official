import jax
import optax
from functools import partial
import jax.numpy as jnp


def mae_loss(pred, target, mask, norm_pix_loss: bool = False):
    if norm_pix_loss:
        mean = target.mean(axis=-1, keepdims=True)
        var = target.var(axis=-1, keepdims=True)
        target = (target - mean) / (var + 1e-6) ** .5
    loss = (pred - target) ** 2
    if mask is not None:
        loss = loss.mean(axis=-1)
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = loss.mean()
    print("loss.shape:", loss.shape)
    return loss


def cross_entropy_loss(logits, labels, smoothing_factor: float = None):
    if smoothing_factor and type(smoothing_factor) == float:
        labels = optax.smooth_labels(labels, alpha=smoothing_factor)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(xentropy)


def binary_xentropy_loss(logits, labels, smoothing_factor: float = None):
    if smoothing_factor and type(smoothing_factor) == float:
        labels = optax.smooth_labels(labels, alpha=smoothing_factor)
    xentropy = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(xentropy)
