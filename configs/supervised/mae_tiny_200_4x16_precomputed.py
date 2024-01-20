import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "mae_tiny"
    config.model.type = "multilabel"
    config.model.num_classes = 527
    config.model.model_args = {
        "mask_ratio": 0.8,
        "img_size": (200, 80),
        "patch_size": (4, 16),
        "decoder_num_heads": 8,
        "decoder_depth": 4,
        "decoder_embed_dim": 384,
        "mode": config.model.type,
        "supervised_globalpool": True,
        "supervised_num_classes": config.model.num_classes
    }
    config.model.pretrained = ""
    config.model.pretrained_fc_only = False
    config.model.patch_embed_args = ml_collections.ConfigDict()

    config.opt = ml_collections.ConfigDict()
    config.opt.optimizer = "Adamw"
    config.opt.learning_rate = 1.5e-5
    config.opt.weight_decay = 1e-3
    config.opt.schedule = "warmupcosine"
    config.opt.warmup_epochs = 10
    config.opt.momentum = 0.9
    config.opt.norm_pix_loss = False
    
    # FC options
    # config.opt.separate_lrs = True
    config.opt.learning_rate = 1e-5
    # config.opt.fc_learning_rate = 1e-3
    config.opt.label_smoothing_factor = 0.01

    config.log_every_steps = 1
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
    config.data.tr_manifest = "/Users/sarthak/Datasets/audioset_logmelspec/meta/balanced.csv"
    config.data.eval_manifest = "/Users/sarthak/Datasets/audioset_logmelspec/meta/eval.csv"
    config.data.tr_samples = 320
    config.data.eval_samples = 320
    config.data.compression = "ZLIB"
    config.data.reader = "spec"
    config.data.cacheable = False
    config.data.jax_transforms = True
    config.data.dataset_name = "audioset"

    config.batch_size = 32
    config.shuffle_buffer_multiplier = 500
    config.precision = "float32"
    config.input_shape = (200, 80, 1)
    config.num_epochs = 100
    config.device = None

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "mw_mae_v2"

    return config
