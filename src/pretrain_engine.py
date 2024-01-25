import jax
import jax.numpy as jnp
import functools
from .helpers.utilities import compute_loss, TrainingMode
from ml_collections import ConfigDict


def train_step(state,
               batch,
               learning_rate_fn,
               cost_func,
               mode=TrainingMode.MAE):
    def loss_fn(params):
        inputs = batch['audio']
        labels = batch['label']
        forwarded_aux, new_model_state = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats, "buffers": state.buffers},
                inputs,
                mutable=['batch_stats'],
                rngs=state.aux_rng_keys
            )
        pred, labels, mask = forwarded_aux[:3]
        loss = cost_func(*forwarded_aux)
        return loss, (new_model_state, pred, labels, mask)
    
    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # grads = jax.lax.pmean(grads, axis_name='batch')   not needed as dynamic_scale already does it
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
        aux, grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')
    
    new_model_state, pred, labels, mask = aux[1]
    loss = aux[0]
    metrics = compute_loss(loss)
    metrics['lr'] = lr
    new_state = state.apply_gradients(grads=grads, 
                                      batch_stats=new_model_state['batch_stats'])
    if dynamic_scale:
        new_state = new_state.replace(
            opt_state=jax.tree_map(
                functools.partial(jnp.where, is_fin), 
                new_state.opt_state, 
                state.opt_state),
            params=jax.tree_map(
                functools.partial(jnp.where, is_fin), 
                new_state.params, 
                state.params)
            )
        metrics['scale'] = dynamic_scale.scale
    return new_state, metrics, mask
