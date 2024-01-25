import flax
from flax.training import checkpoints


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir, keep=3, keep_every_n_steps=None):
    state = flax.jax_utils.unreplicate(state)
    step = int(state.step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, 
                                             keep=keep, keep_every_n_steps=keep_every_n_steps)

def save_best_checkpoint(state, workdir, best_acc):
    state = flax.jax_utils.unreplicate(state)
    step = int(state.step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, prefix="best_ckpt_", keep=3)