import copy
import time
import wandb
import jax
import jax.numpy as jnp
import functools
from clu import metric_writers
from clu import periodic_actions
from flax.training.common_utils import get_metrics
import flax.jax_utils as flax_jax_utils
from .helpers import utilities, optimization, trainstate, initialization, checkpointing
from .data import prepare_datasets_v2
from .loss import mae_loss, cross_entropy_loss, binary_xentropy_loss
from .modules import create_model
from . import supervised_engine
from . import pretrain_engine
import logging


class Trainer(object):
    def __init__(self, config, workdir, no_wandb, seed=0, inference=False):
        if not no_wandb:
            self.wandb_logger = wandb.init(
                project='{}'.format(config.wandb.get("project", "audax-supervised")),
                group="{}".format(config.data.dataset_name),
                config=config.to_dict(), 
                name=workdir.split("/")[-1])
        else:
            self.wandb_logger = None
        self.NUM_PROCESSES = jax.process_count()
        self.DEVICE_COUNT = jax.device_count()
        self.WORKER_ID = jax.process_index()
        self.writer = metric_writers.create_default_writer(logdir=workdir, just_logging=jax.process_index() != 0)
        utilities.write_config_to_json(workdir, config)
        self.rng = jax.random.PRNGKey(seed)
        if config.batch_size % jax.device_count() > 0:
            raise ValueError(f'Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}')
        self.local_batch_size = config.batch_size // self.NUM_PROCESSES
        logging.info("Process count: {}".format(jax.process_count()))
        device = config.get("device", None)
        if device is not None:
            self.local_devices = [jax.local_devices()[device]]
        else:
            self.local_devices = jax.local_devices()
        self.global_device = jax.devices()
        
        self.mode = utilities.TrainingMode(config.model.type)
        if config.data.get("mean_stats", None):
            mean = jnp.load(config.data.mean_stats)
            std = jnp.load(config.data.std_stats)
        else:
            mean = None
            std = None
        self.input_dtype = utilities.get_dtype(config.precision)
        self.p_feature_extract_fn = jax.pmap(
            functools.partial(
                utilities.precomputed_feature_extract_fn,
                dtype=self.input_dtype,
                mean=mean, std=std
            ),
            axis_name='batch', devices=self.local_devices
        )
        self.config = config
        self.workdir = workdir
        self.inference = inference
        self._optimization_setup_done = False
        self._data_setup_done = False
        self._model_setup_done = False

        self.setup_optimization()
        if self.inference:
            self.prepare_inference_state()
        else:
            self.prepare_model_state()

    def setup_data(self, train_only=False):
        if self._data_setup_done:
            return
        if self.NUM_PROCESSES == 1:
            train_iter, eval_iter = prepare_datasets_v2(self.config, self.local_batch_size, input_dtype=self.input_dtype)
        else:
            train_iter, eval_iter = prepare_datasets_v2(self.config, self.local_batch_size, input_dtype=self.input_dtype,
                                                        num_shards=self.NUM_PROCESSES, worker_index=self.WORKER_ID)
        self.train_iter = utilities.create_input_iter(train_iter, devices=self.local_devices)
        if not train_only:
            self.eval_iter = utilities.create_input_iter(eval_iter, devices=self.local_devices)
        else:
            self.eval_iter = None
        self._data_setup_done = True

    def setup_optimization(self):
        if self._optimization_setup_done:
            return
        num_examples = self.config.data.tr_samples
        if self.config.get("steps_per_epoch", -1) == -1:
            self.steps_per_epoch = (num_examples // self.config.batch_size)
        else:
            self.steps_per_epoch = self.config.get("steps_per_epoch")
        if self.config.num_train_steps == -1:
            self.num_steps = int(self.steps_per_epoch * self.config.num_epochs)
            self.num_epochs = self.config.num_epochs
        else:
            self.num_steps = self.config.num_train_steps
            self.num_epochs = self.config.num_train_steps // self.steps_per_epoch
        logging.info("num_steps: {} | num_epochs: {} | steps_per_epoch: {}".format(self.num_steps, self.num_epochs, self.steps_per_epoch))
        if self.config.steps_per_eval == -1:
            num_validation_examples = self.config.data.eval_samples
            self.steps_per_eval = num_validation_examples // self.config.batch_size
        else:
            self.steps_per_eval = self.config.steps_per_eval
        self.steps_per_checkpoint = self.steps_per_epoch
        base_learning_rate = self.config.opt.get("grad_accum_steps", 1) * self.config.opt.learning_rate * self.config.batch_size / 256.
        logging.info("Base learning rate: {}".format(base_learning_rate))
        self.learning_rate_fn = optimization.create_learning_rate_fn(
            self.config, 
            base_learning_rate, 
            self.steps_per_epoch, 
            num_epochs=self.num_epochs
        )
        self._optimization_setup_done = True

    def prepare_model_state(self):
        raise NotImplementedError("prepare_model_state() not implemented")

    def prepare_inference_state(self):
        raise NotImplementedError("prepare_inference_state() not implemented")

    def fit(self):
        if not self._data_setup_done:
            self.setup_data()
        assert self._optimization_setup_done, "Optimization not setup"
        assert self._model_setup_done, "Model not setup"
        train_metrics = []
        hooks = []
        if jax.process_index() == 0:
            hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=self.workdir)]
        train_metrics_last_t = time.time()
        logging.info('Initial compilation, this might take some minutes...')
        step_offset = self.step_offset
        for step, batch in zip(range(step_offset, self.num_steps), self.train_iter):
            if self.p_feature_extract_fn:
                batch['audio'] = self.p_feature_extract_fn(batch['audio'])
            if step == 0:
                print(f"batch['audio'] info {batch['audio'].shape}, {batch['audio'].dtype}")
            self.state, metrics, auxiliary = self.p_train_step(self.state, batch)
            for h in hooks:
                h(step)
            if step == step_offset:
                logging.info('Initial compilation completed.')
            train_metrics.append(metrics)
            if (step + 1) % self.config.log_every_steps == 0:
                train_metrics = get_metrics(train_metrics)
                summary = {
                    f'train_{k}': v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                summary['steps_per_second'] = self.config.log_every_steps / (
                        time.time() - train_metrics_last_t)
                self.writer.write_scalars(step + 1, copy.copy(summary))
                if self.wandb_logger:
                    self.wandb_logger.log(summary, step + 1)
                train_metrics = []
                train_metrics_last_t = time.time()

            if (step + 1) % self.steps_per_epoch == 0:
                self.eval(step)
            
            if (step + 1) % self.steps_per_checkpoint == 0 or step + 1 == self.num_steps:
                # self.state = state
                self.ckpt_on_epoch_end(self.state)

            self.writer.flush()

        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if self.wandb_logger:
            self.wandb_logger.finish()
        return self.state

    def ckpt_on_epoch_end(self, state):
        self.state = utilities.sync_batch_stats(state)
        checkpointing.save_checkpoint(self.state, 
                                  self.workdir, 
                                  keep=3, 
                                  keep_every_n_steps=self.steps_per_checkpoint*10)

    def eval(self, step):
        raise NotImplementedError("eval() not implemented")


class MAETrainer(Trainer):
    def __init__(self, config, workdir, no_wandb, seed=0, inference=False):
        super().__init__(config, workdir, no_wandb, seed, inference)
    
    def setup_data(self, train_only=True):
        return super().setup_data(train_only)

    def prepare_model_state(self):
        assert self._optimization_setup_done, "Optimization not setup"
        assert self.mode == utilities.TrainingMode.MAE
        cost_fn = functools.partial(mae_loss, norm_pix_loss=self.config.opt.get("norm_pix_loss", False))
        model = create_model(self.config, precision=self.config.precision)
        logging.info(str(model))
        state = initialization.create_train_state(self.rng, 
                            self.config,
                            model,
                            self.learning_rate_fn,
                            additional_aux_rngs=['random_masking', "additive_noise", "gumbel"])
        state = checkpointing.restore_checkpoint(state, self.workdir)
        self.step_offset = int(state.step)
        self.state = flax_jax_utils.replicate(state)
        logging.info('Train state ready...')
        self.p_train_step = jax.pmap(
            functools.partial(pretrain_engine.train_step,
                              learning_rate_fn=self.learning_rate_fn,
                              cost_func=cost_fn,
                              mode=self.mode),
            axis_name='batch')
        self.model = model
        self._model_setup_done = True
    
    def prepare_inference_state(self):
        assert self._optimization_setup_done, "Optimization not setup"
        assert self.mode == utilities.TrainingMode.MAE
        model = create_model(self.config, precision=self.config.precision)
        logging.info(str(model))
        state = initialization.create_train_state(self.rng, 
                            self.config,
                            model,
                            self.learning_rate_fn,
                            additional_aux_rngs=['random_masking', "additive_noise", "gumbel"])
        state = checkpointing.restore_checkpoint(state, self.workdir)
        self.step_offset = int(state.step)
        self.state = state
        self.model = model
        self._model_setup_done = True

    def eval(self, step):
        return


class SupervisedTrainer(Trainer):
    def __init__(self, config, workdir, no_wandb, seed=0):
        super().__init__(config, workdir, no_wandb, seed)
        self.best_val_acc = 0.0
        from sklearn.metrics import average_precision_score
        self._avg_prec_func = average_precision_score
    
    def setup_data(self, train_only=False):
        return super().setup_data(train_only)
    
    def prepare_model_state(self):
        assert self._optimization_setup_done, "Optimization not setup"
        assert self.mode == utilities.TrainingMode.MULTICLASS or self.mode == utilities.TrainingMode.MULTILABEL
        label_smoothing_factor = self.config.opt.get("label_smoothing_factor", None)
        if self.mode == utilities.TrainingMode.MULTICLASS:
            cost_fn = functools.partial(cross_entropy_loss, smoothing_factor=label_smoothing_factor)
        else:
            cost_fn = functools.partial(binary_xentropy_loss,  smoothing_factor=label_smoothing_factor)
        mixup_alpha = self.config.opt.get("mixup_alpha", 0.0)
        if mixup_alpha != 0.:
            raise NotImplementedError("Mixup not implemented")
        else:
            mixup_func = None
            mixup_criterion_func = None
        model = create_model(self.config, precision=self.config.precision)
        logging.info(str(model))
        fc_learning_rate = self.config.opt.get("grad_accum_steps", 1) * self.config.opt.get("fc_learning_rate", 1e-3) * self.config.batch_size / 256.
        separate_lrs = self.config.opt.get("separate_lrs", False)
        if separate_lrs:
            fc_lr_fn = optimization.create_learning_rate_fn(self.config, 
                                                            fc_learning_rate, 
                                                            self.steps_per_epoch, 
                                                            num_epochs=self.num_epochs)
        else:
            fc_lr_fn = None
        if self.config.model.get("pretrained", None):
            state = initialization.create_finetune_train_state_from_pretrained(self.rng, self.config,
                                                                        model, self.learning_rate_fn,
                                                                        self.config.model.pretrained,
                                                                        self.config.model.get("pretrained_prefix",
                                                                                        "checkpoint_"),
                                                                        copy_all=True,
                                                                        to_copy=[],
                                                                        fc_only=self.config.model.get("pretrained_fc_only",
                                                                                                False),
                                                                        fc_learning_rate_fn=fc_lr_fn,
                                                                        separate_lrs=fc_lr_fn,
                                                                        additional_aux_rngs=['random_masking'],
                                                                        # apply_fn_override=model.forward_features
                                                                        )
            logging.info('Train state created ...')

        else:
            state = initialization.create_train_state(self.rng, self.config, model, 
                                                      self.learning_rate_fn, 
                                                      additional_aux_rngs=['random_masking'],
                                                    #   apply_fn_override=model.forward_features
                                                      )
        state = checkpointing.restore_checkpoint(state, self.workdir)
        self.step_offset = int(state.step)
        self.state = flax_jax_utils.replicate(state)
        logging.info('Train state ready...')
        self.p_train_step = jax.pmap(
            functools.partial(supervised_engine.train_step, 
                              learning_rate_fn=[self.learning_rate_fn, fc_lr_fn],
                              cost_func=cost_fn,
                              mode=self.mode, mixup_func=mixup_func, 
                              mixup_criterion_func=mixup_criterion_func),
            axis_name='batch'
        )
        self.p_eval_step = jax.pmap(
            functools.partial(supervised_engine.eval_step,
                              mode=self.mode,
                              cost_func=cost_fn),
            axis_name='batch'
        )
        self.model = model
        self._model_setup_done = True
    
    def eval(self, step):
        epoch = step // self.steps_per_epoch
        eval_metrics = []
        eval_logits = []
        eval_labels = []
        self.state = utilities.sync_batch_stats(self.state)

        for _ in range(self.steps_per_eval):
            eval_batch = next(self.eval_iter)
            if self.p_feature_extract_fn:
                eval_batch['audio'] = self.p_feature_extract_fn(eval_batch['audio'])
            metrics, logits, labels = self.p_eval_step(self.state, eval_batch)
            eval_metrics.append(metrics)
            eval_logits.append(logits)
            eval_labels.append(labels)
        
        eval_metrics = get_metrics(eval_metrics)
        summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
        if self.mode == utilities.TrainingMode.MULTILABEL:
            eval_logits = jnp.concatenate([jax.device_get(x) for x in eval_logits])
            eval_labels = jnp.concatenate([jax.device_get(x) for x in eval_labels])
            eval_logits = eval_logits.reshape(-1, eval_logits.shape[-1])
            eval_labels = eval_labels.reshape(-1, eval_labels.shape[-1])
            map_value = self._avg_prec_func(eval_labels.astype('float32'), eval_logits.astype('float32'),
                                            average="macro")
            summary['accuracy'] = map_value
        logging.info('eval epoch: %d, loss: %.4f, accuracy: %.4f',
                         epoch, summary['loss'], summary['accuracy'] * 100)
        
        if summary['accuracy'] >= self.best_val_acc:
            self.best_val_acc = summary['accuracy']
            checkpointing.save_best_checkpoint(self.state, self.workdir, self.best_val_acc)
        self.writer.write_scalars(
            step + 1,
            {f'eval_{key}': val for key, val in summary.items()}
        )
        if self.wandb_logger:
            self.wandb_logger.log({f'eval_{key}': val for key, val in summary.items()}, step + 1)
