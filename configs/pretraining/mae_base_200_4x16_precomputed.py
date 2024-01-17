import ml_collections
import os
DATASETS_DIR = os.environ.get("DATASETS_BASE_DIR", "")


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "mae_base"
    config.model.type = "mae"
    config.model.num_classes = 1000     # needed for dataset parsing. Is not used.
    # config.model.img_size = 40000
    # config.model.patch_size = 16
    # config.model.in_chans = 1
    # config.model.embed_dim = 768
    config.model.model_args = {
        "mask_ratio": 0.8,
        "img_size": (200, 80),
        "patch_size": (4, 16),
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
    # config.opt.grad_accum_steps = 1
    # config.opt.agc_clip_val = 0.01
    # config.opt.opt_func = "mae_infoNCE"
    # config.opt.contrastive_temp = 0.1
    # config.opt.alpha = 1.

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
    config.data.tr_samples = 2032320
    config.data.eval_manifest = os.path.join(DATASETS_DIR, "audioset_logmelspec/meta/eval.csv")
    config.data.eval_samples = 17408
    # config.data.mean_stats = "/home/sarthak/my_disk/Datasets/audioset_logmelspec/mean.npy"
    # config.data.std_stats = "/home/sarthak/my_disk/Datasets/audioset_logmelspec/std.npy"
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
