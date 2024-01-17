import jax
import jax.numpy as jnp
import collections
from itertools import repeat


def unbatched_gather(x, ids_keep):
    return x[ids_keep, Ellipsis]


batched_gather = jax.vmap(unbatched_gather)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
to_1tuple = _ntuple(1)


def constant_init(key, shape, dtype=jnp.float_, constant=0.04):
    return jnp.ones(shape, jax.dtypes.canonicalize_dtype(dtype)) * constant