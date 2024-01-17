import optax
from ml_collections import ConfigDict
import logging


def step_lr(base_lr, reduce_every_n_steps, total_steps, alpha=0.5):
    schedules = []
    boundaries = []
    curr_lr = base_lr
    for step in range(1, total_steps+1, reduce_every_n_steps):
        boundaries.append(step-1)
        schedules.append(optax.constant_schedule(curr_lr))
        curr_lr *= alpha
    boundaries = boundaries[1:]
    schedule_fn = optax.join_schedules(
        schedules=schedules,
        boundaries=boundaries
    )
    return schedule_fn


def create_learning_rate_fn(
        config: ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int,
        num_epochs: int=None):
    """Create learning rate schedule."""
    if config.opt.schedule == "warmupcosine":
        logging.info("Using cosine learning rate decay")
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=base_learning_rate,
            transition_steps=config.opt.warmup_epochs * steps_per_epoch)
        if num_epochs is None:
            num_epochs = config.num_epochs
        cosine_epochs = max(num_epochs - config.opt.warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[config.opt.warmup_epochs * steps_per_epoch])
    elif config.opt.schedule == "cosine_decay":
        cosine_epochs = num_epochs
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = cosine_fn
    elif config.opt.schedule == "exp_decay":
        warmup_epochs = int(config.opt.get("warmup_epochs", 0))
        transition_steps = (num_epochs - warmup_epochs) * steps_per_epoch
        fixed_steps = warmup_epochs * steps_per_epoch
        schedule_fn = optax.exponential_decay(
            init_value=base_learning_rate,
            decay_rate=config.opt.get("decay_rate", 0.1),
            transition_begin=fixed_steps+1,
            transition_steps=transition_steps,
            end_value=config.opt.get("end_value", 1e-7)
        )
    elif config.opt.schedule == "step":
        lr_step_every = int(config.opt.get("step_epochs", 10) * steps_per_epoch)
        alpha = config.opt.get("alpha", 0.5)
        total_steps = steps_per_epoch * num_epochs
        schedule_fn = step_lr(base_learning_rate, 
                              reduce_every_n_steps=lr_step_every,
                              total_steps=total_steps,
                              alpha=alpha)
    else:
        schedule_fn = base_learning_rate
    return schedule_fn


def create_optimizer(config: ConfigDict, learning_rate_fn):
    optimizer_name = config.opt.optimizer.lower()
    if optimizer_name == "adamw":
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=config.opt.weight_decay
        )
    elif optimizer_name == "lars":
        tx = optax.lars(
            learning_rate=learning_rate_fn,
            weight_decay=config.opt.weight_decay
        )
    elif optimizer_name == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.opt.get("momentum", 0.9),
            nesterov=config.opt.get("nesterov", False)
        )
    else:
        raise ValueError("optimizer {} not supported. Valid values are [adamw, lars, sgd]")
    return tx
