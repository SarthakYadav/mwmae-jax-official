# Written for audax by / Copyright 2022, Sarthak Yadav
import os
from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
import flax
from ml_collections import config_flags
import tensorflow as tf
# from src import train_mae, train_supervised
# from src import eval_supervised
from src.trainer import MAETrainer, SupervisedTrainer
# from audax.training_utils import train_supervised, eval_supervised
# logging.set_verbosity(logging.ERROR)
# logging.get_absl_handler().python_handler.use_absl_log_file(log_dir='./logs')
# print('Logging to: %s' % logging.get_log_file_name())
flax.config.update('flax_use_orbax_checkpointing', False)

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_string('mode', "ssl", 'Mode (Default: ssl, Options: [ssl, train, eval])')
flags.DEFINE_bool("no_wandb", False, "To switch off wandb_logging")
flags.DEFINE_bool("strided_eval", False, "To switch on strided_eval")
flags.DEFINE_string("pretrained_dir", None, "Directory of the pretrained SSL model")
flags.DEFINE_integer("seed", 0, "seed")
flags.DEFINE_string("eval_manifest_override", None, "Path to eval manifest")
flags.DEFINE_integer("eval_steps_override", None, "Number of samples in overriden eval manifest")
flags.DEFINE_float("eval_duration", None, "Duration of eval signal")
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                           f'process_count: {jax.process_count()}')
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                         FLAGS.workdir, 'workdir')

    if FLAGS.mode == "ssl":
        # train_mae.train(FLAGS.config, FLAGS.workdir, FLAGS.no_wandb, FLAGS.seed)
        trainer = MAETrainer(FLAGS.config, FLAGS.workdir, FLAGS.no_wandb, FLAGS.seed)
        trainer.fit()
    elif FLAGS.mode == "train":
        if FLAGS.pretrained_dir != None and os.path.exists(FLAGS.pretrained_dir):
            existing_pretrained_dir = FLAGS.config.model.get("pretrained", None)
            if existing_pretrained_dir is not None:
                logging.info("Overriding pretrained dir {} to {}".format(existing_pretrained_dir, FLAGS.pretrained_dir))
            FLAGS.config.model.pretrained = FLAGS.pretrained_dir
        # train_supervised.train_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.no_wandb, FLAGS.seed)
        trainer = SupervisedTrainer(FLAGS.config, FLAGS.workdir, FLAGS.no_wandb, FLAGS.seed)
        trainer.fit()
    # elif FLAGS.mode == "eval":
    #     if FLAGS.eval_duration is None:
    #         raise ValueError("with FLAGS.mode = eval, eval_duration has to be provided.")
    #     eval_dur = FLAGS.eval_duration
    #     eval_supervised.evaluate(FLAGS.workdir, eval_dur, 
    #                              eval_manifest_override=FLAGS.eval_manifest_override,
    #                              eval_steps_override=FLAGS.eval_steps_override,
    #                              strided_eval=FLAGS.strided_eval)
    else:
        raise ValueError(f"Unsupported FLAGS.training_mode: {FLAGS.mode}. Supported are ['pretrain', 'train', 'eval']")


if __name__ == '__main__':
    flags.mark_flags_as_required(['workdir'])
    app.run(main)
