import jax
import functools
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, Union, List
from .helpers.utilities import compute_loss, compute_accuracy, TrainingMode


def train_step(state,
               batch,
               learning_rate_fn: Union[List[Callable], Callable, float],
               cost_func,
               mixup_func=None,
               mixup_criterion_func=None,
               mode=TrainingMode.MULTICLASS):
    inputs = batch['audio']
    targets = batch['label']
    if mixup_func is not None:
        inputs, y_a, y_b, lam = mixup_func(state.aux_rng_keys["mixup"], inputs, targets)
    
    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {
                'params': {**params, **state.frozen_params},
                'batch_stats': state.batch_stats, 
                "buffers": state.buffers
            },
            inputs,
            mutable=['batch_stats'],
            rngs=state.aux_rng_keys
        )
        if mixup_func is not None:
            loss = mixup_criterion_func(cost_func, logits, y_a, y_b, lam)
        else:
            loss = cost_func(logits, targets)
        return loss, (new_model_state, logits)
    
    step = state.step
    dynamic_scale = state.dynamic_scale
    
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # grads = jax.lax.pmean(grads, axis_name='batch')   not needed as dynamic_scale already does it
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
        aux, grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')
    new_model_state, logits = aux[1]
    loss = aux[0]
    metrics = compute_loss(loss)
    if not isinstance(learning_rate_fn, list):
        learning_rate_fn = [learning_rate_fn]
    for ix in range(len(learning_rate_fn)):
        lr_fn = learning_rate_fn[ix]
        if lr_fn is None:
            continue
        elif isinstance(lr_fn, Callable):
            lr = lr_fn(step)
        else:
            lr = lr_fn
        metrics[f'learning_rate_{ix}'] = lr
    if mode == TrainingMode.MULTICLASS:
        acc = compute_accuracy(logits, targets)
        metrics.update(acc)
    new_state = state.apply_gradients(grads=grads, 
                                      batch_stats=new_model_state['batch_stats']
                                     )
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

    return new_state, metrics, None


def eval_step(state, batch, cost_func, mode=TrainingMode.MULTICLASS):
    variables = {
        'params': state.get_all_params,                    # absolutely ok to just use state.get_all_params here
        'batch_stats': state.batch_stats,
        "buffers": state.buffers
    }
    logits = state.apply_fn(
        variables, batch['audio'], train=False, mutable=False)
    metrics = compute_loss(cost_func(logits, batch['label']))

    if mode == TrainingMode.MULTICLASS:
        acc = compute_accuracy(logits, batch['label'])
        metrics.update(acc)
    
    return metrics, logits, batch['label']
