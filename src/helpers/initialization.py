import logging
import jax
import flax
import jax.numpy as jnp
from .. import modules
from .trainstate import TrainState_v2
from ml_collections import ConfigDict
from flax.training.dynamic_scale import DynamicScale
from flax.training import checkpoints
import optax
from .optimization import create_optimizer
from .utilities import get_dtype


# DEFAULT_AUX_RNG_KEYS = ["random_masking", "dropout", "drop_path"]
DEFAULT_AUX_RNG_KEYS = ["dropout", "drop_path", "mixup", "spec_aug"]


def initialize(key, inp_shape, model, 
               aux_rng_keys=DEFAULT_AUX_RNG_KEYS):
    input_shape = (2,) + inp_shape

    @jax.jit
    def init(*args):
        return model.init(*args)
    num_keys = len(aux_rng_keys)
    key, *subkeys = jax.random.split(key, num_keys+1)
    rng_keys = {aux_rng_keys[ix]: subkeys[ix] for ix in range(len(aux_rng_keys))}
    variables = init({'params': key, **rng_keys}, 
                     jnp.ones(input_shape, model.dtype))
    rngs = flax.core.FrozenDict(rng_keys)
    has_batch_stats = False
    has_buffers = False
    if "batch_stats" in variables.keys():
        has_batch_stats = True
    if "buffers" in variables.keys():
        has_buffers = True
    res = (
        variables['params'],
        variables['batch_stats'] if has_batch_stats else flax.core.freeze({}),
        variables['buffers'] if has_buffers else flax.core.freeze({}),
        rngs
    )
    return res


def create_train_state(rng, 
                       config: ConfigDict,
                       model, 
                       learning_rate_fn, 
                       additional_aux_rngs=[],
                       apply_fn_override=None):
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.precision == "float16" and platform == 'gpu':
        dynamic_scale = DynamicScale()
    else:
        dynamic_scale = None
    if additional_aux_rngs is not None and len(additional_aux_rngs) != 0:
        aux_rngs = DEFAULT_AUX_RNG_KEYS + additional_aux_rngs
    else:
        aux_rngs = DEFAULT_AUX_RNG_KEYS
    params, batch_stats, buffers, rng_keys = initialize(rng, config.input_shape, model, aux_rng_keys=aux_rngs)
    tx = create_optimizer(config, learning_rate_fn)
    agc_clip_val = config.opt.get("agc_clip_val", None)
    if agc_clip_val is not None:
        logging.info(f"Using adaptive gradient clipping with clip factor {agc_clip_val}")
        agc = optax.adaptive_grad_clip(clipping=agc_clip_val)
        tx = optax.chain(
            tx,
            agc
        )
    apply_fn = apply_fn_override if apply_fn_override is not None else model.apply
    state = TrainState_v2.create(
        apply_fn=apply_fn,
        params=params,
        frozen_params=flax.core.freeze({}),
        tx=tx,
        batch_stats=batch_stats,
        buffers=buffers,
        aux_rng_keys=rng_keys,
        dynamic_scale=dynamic_scale)
    return state


def create_finetune_train_state_from_pretrained(rng, config: ConfigDict,
                                                model, learning_rate_fn,
                                                pretrained_work_dir, 
                                                pretrained_prefix="checkpoint_",
                                                copy_all=False,
                                                to_copy=['encoder'],
                                                fc_only=False,
                                                fc_learning_rate_fn=None,
                                                separate_lrs=True,
                                                additional_aux_rngs=[],
                                                apply_fn_override=None):
    logging.info("Making train state from pretrained..")
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.precision == "float16" and platform == 'gpu':
        dynamic_scale = DynamicScale()
    else:
        dynamic_scale = None
    if additional_aux_rngs is not None and len(additional_aux_rngs) != 0:
        aux_rngs = DEFAULT_AUX_RNG_KEYS + additional_aux_rngs
    else:
        aux_rngs = DEFAULT_AUX_RNG_KEYS
    params, batch_stats, buffers, rng_keys = initialize(rng, config.input_shape, model, aux_rng_keys=aux_rngs)

    # load pretrained ckpt to a dictionary
    pretrained_state_dict = checkpoints.restore_checkpoint(pretrained_work_dir, None,
                                                           prefix=pretrained_prefix)
    pretrained_params = pretrained_state_dict['params']
    pretrained_batch_stats = pretrained_state_dict['batch_stats']
    pretrained_buffers = pretrained_state_dict['buffers']

    # unfreeze classifier params and batch_stats
    params = flax.core.unfreeze(params)
    batch_stats = flax.core.unfreeze(batch_stats)
    buffers = flax.core.unfreeze(buffers)
    if copy_all:
        logging.info("copy_all is True. Attempting to copy all parameters by name")
        to_copy = list(set(params['model'].keys()).intersection(set(pretrained_params.keys())))
    logging.info("Copying the following parameters: \n[{}]".format(",".join(to_copy)))

    for k in to_copy:
        assert k in params['model'].keys()
        params['model'][k] = pretrained_params[k]
        try:
            batch_stats['model'][k] = pretrained_batch_stats[k]
        except KeyError as ex:
            pass
        except TypeError as ex:
            pass
        try:
            buffers['model'][k] = pretrained_buffers[k]
        except KeyError as ex:
            pass
        except TypeError as ex:
            pass
    
    if fc_only and separate_lrs:
        raise ValueError("Both fc_only and separate_lrs cannot be True")

    if fc_only:
        logging.info("Training fc only")
        fc_tx = create_optimizer(config, learning_rate_fn)
        partition_optimizers = {'features': optax.set_to_zero(), 'final_head': fc_tx}
    
    if separate_lrs:
        logging.info("making separate optimizers for fc and model")
        fc_tx = create_optimizer(config, fc_learning_rate_fn)
        model_tx = create_optimizer(config, learning_rate_fn)
        partition_optimizers = {'features': model_tx, 'final_head': fc_tx}
    
    
    if not fc_only and not separate_lrs:
        logging.info("Model and FC have same optimizer..")
        tx = create_optimizer(config, learning_rate_fn)
    else:
        param_partitions = flax.core.freeze(flax.traverse_util.path_aware_map(
            lambda path, v: 'features' if 'model' in path else 'final_head', params))
        tx = optax.multi_transform(partition_optimizers, param_partitions)
    

    trainable_params = flax.core.freeze(params)
    batch_stats = flax.core.freeze(batch_stats)
    buffers = flax.core.freeze(buffers)
    frozen_params = flax.core.freeze({})
    # make the train state now
    
    grad_accum_steps = config.opt.get("grad_accum_steps", 1)
    if grad_accum_steps > 1:
        logging.info("Using gradient accumulation of {} steps".format(grad_accum_steps))
        tx = optax.MultiSteps(tx, grad_accum_steps)
    if apply_fn_override is not None:
        logging.info("Initializing trainstate in apply_fn_override.")
        state = TrainState_v2.create(
        apply_fn=apply_fn_override,
        params=trainable_params,
        frozen_params=frozen_params,
        tx=tx,
        batch_stats=batch_stats,
        buffers=buffers,
        aux_rng_keys=rng_keys,
        dynamic_scale=dynamic_scale)
        logging.info("Training state created in apply_fn_override.")
        
    else:    
        logging.info("Initializing trainstate.")    
        state = TrainState_v2.create(
            apply_fn=model.apply,
            params=trainable_params,
            frozen_params=frozen_params,
            tx=tx,
            batch_stats=batch_stats,
            buffers=buffers,
            aux_rng_keys=rng_keys,
            dynamic_scale=dynamic_scale)
        logging.info("Training state created.")
    return state
