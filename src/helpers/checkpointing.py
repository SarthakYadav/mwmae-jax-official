import jax
from flax.training import checkpoints


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir, keep=3, keep_every_n_steps=None):
    # if jax.process_index() == 0:
    #     # get train state from the first replica
    #     state = jax.device_get(jax.tree_map(lambda x:x[0], state))
    #     step = int(state.step)
    #     checkpoints.save_checkpoint(workdir, state, step, keep=3)
    state = jax.device_get(jax.tree_map(lambda x:x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, 
                                             keep=keep, keep_every_n_steps=keep_every_n_steps)

def save_best_checkpoint(state, workdir, best_acc):
    # if jax.process_index() == 0:
    #     # get train state from the first replica
    #     state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    #     checkpoints.save_checkpoint(workdir, state, step, prefix="best_ckpt_", keep=3)
    state = jax.device_get(jax.tree_map(lambda x:x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, prefix="best_ckpt_", keep=3)