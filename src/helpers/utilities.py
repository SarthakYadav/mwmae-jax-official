import os
import json
import jax
import flax
import enum
import logging
import functools
from jax import lax
import jax.numpy as jnp


@enum.unique
class TrainingMode(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    CONTRASTIVE = "contrastive"
    COLA = "cola"
    MAE = "mae"


def precomputed_feature_extract_fn(x, dtype=jnp.float32, mean=None, std=None):
    x = x[Ellipsis, jnp.newaxis]
    if mean is not None and std is not None:
        print("Normalizing!!!!!!!!!! x with shape:", x.shape)
        x = (x - mean) / std
    x = x.astype(dtype)
    return x

# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def get_dtype(precision):
    platform = jax.local_devices()[0].platform
    if platform == "tpu":
        model_dtype = jnp.bfloat16 if "16" in precision else jnp.float32
    else:
        if precision == "float16":
            model_dtype = jnp.float16
        elif precision == "bfloat16":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float32
    return model_dtype


def compute_loss(loss):
    metrics = {
        'loss': loss
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def compute_accuracy(logits, labels):    
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    metrics = {
        'accuracy': accuracy
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def prepare_tf_data(xs, devices=None):
    """Convert a input batch from tf Tensors to numpy arrays."""
    if devices is None:
        local_device_count = jax.local_device_count()
    elif type(devices) == list or type(devices) == tuple:
        local_device_count = len(devices)
    else:
        raise ValueError("Devices should either be None or a list of jax.device")

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, channel) to
        # (local_devices, device_batch_size, height, width, channel)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def write_config_to_json(workdir, config):
    config_path = os.path.join(workdir, "config.json")
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    if os.path.exists(config_path):
        logging.info(f"config file {config_path} exists.. Not overwriting.")
        return
    with open(config_path, "w") as fd:
        json.dump(config.to_dict(), fd)


def create_input_iter(ds, devices=None):
    ds = ds.repeat()
    prep_data = functools.partial(prepare_tf_data, devices=devices)
    it = map(prep_data, ds)
    it = flax.jax_utils.prefetch_to_device(it, 10, devices=devices)
    return it


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    if len(state.batch_stats) != 0:
        return state.replace(batch_stats=cross_replica_mean(state.batch_stats))
    else:
        return state
