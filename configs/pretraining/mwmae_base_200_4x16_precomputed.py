import ml_collections
import os
DATASETS_DIR = os.environ.get("DATASETS_BASE_DIR", "")


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "mae_base"
    config.model.type = "mae"
    config.model.num_classes = 1000     # needed for dataset parsing. Is not used.
    config.model.model_args = {
        "mask_ratio": 0.8,
        "img_size": (200, 80),
        "patch_size": (4, 16),
        "window_sizes": 0,          # use optimized blocks for encoder
        "decoder_window_sizes": [2, 5, 10, 25, 50, 125, 0, 0],
        "decoder_num_heads": 8,
        "decoder_depth": 4,
        "decoder_embed_dim": 384
    }
    config.model.patch_embed_args = ml_collections.ConfigDict()

    config.opt = ml_collections.ConfigDict()
    config.opt.optimizer = "Adamw"
    config.opt.learning_rate = 1.5e-5
    config.opt.weight_decay = 0.05
    config.opt.schedule = "warmupcosine"
    config.opt.warmup_epochs = 10
    config.opt.momentum = 0.9
    config.opt.norm_pix_loss = False

    config.log_every_steps = 100
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.audio_config = ml_collections.ConfigDict()
    config.audio_config.normalize_spec = True
    config.audio_config.sample_rate = 16000
    config.audio_config.features = "precomputed"
    config.audio_config.min_duration = None  # min duration in seconds
    # config.audio_config.num_freqs = 200
    config.audio_config.timesteps = 200

    config.data = ml_collections.ConfigDict()
    config.data.tr_manifest = os.path.join(DATASETS_DIR, "audioset_logmelspec/meta/unbalanced_full.csv")
    config.data.eval_manifest = os.path.join(DATASETS_DIR, "audioset_logmelspec/meta/eval.csv")
    config.data.tr_samples = 2032320
    config.data.eval_samples = 17408
    config.data.compression = "ZLIB"
    config.data.reader = "spec"
    config.data.cacheable = False
    config.data.jax_transforms = True
    config.data.dataset_name = "audioset"

    config.batch_size = 8*128
    config.shuffle_buffer_multiplier = 500
    config.precision = "bfloat16"
    config.input_shape = (200, 80, 1)
    config.num_epochs = 100
    config.device = None

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "mw_mae_v2"

    return config
