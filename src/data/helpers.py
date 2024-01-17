"""
Helper utilities for parsing experiment configs and creating data splits.

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import ml_collections
import pandas as pd
import functools
from . import dataset
from . import transforms


def get_tfrecord_parser(config: ml_collections.ConfigDict, cropper,
                        label_parser_func, is_val=False):
    if config.audio_config.min_duration is not None:
        desired_seq_len = int(config.audio_config.sample_rate * config.audio_config.min_duration)
    else:
        desired_seq_len = int(config.audio_config.sample_rate * 1.)
    if config.audio_config.get("maxabs_norm", False):
        print("USING maxabs_norm")
    if config.data.reader == "spec":
        flip_ft = config.audio_config.get("flip_ft_axis", False)
        print("!!!!!!!FLIP_FT:", flip_ft)
        parser_fn = functools.partial(
            dataset.parse_tfrecord_specgram,
            label_parser=label_parser_func,
            cropper=cropper,
            normalize_spec=config.audio_config.get("normalize_spec", True),
            flip_ft_axis=config.audio_config.get("flip_ft_axis", False)
        )
    else:
        parser_fn = functools.partial(dataset.parse_tfrecord_fn_v2,
                                      label_parser=label_parser_func,
                                      cropper=cropper,
                                      seg_length=desired_seq_len)
    return parser_fn


def prepare_datasets_v2(config: ml_collections.ConfigDict, batch_size, input_dtype, num_shards=None, worker_index=None):
    train_files = pd.read_csv(config.data.tr_manifest)['files'].values
    val_files = pd.read_csv(config.data.eval_manifest)['files'].values

    if config.audio_config.min_duration is not None:
        desired_seq_len = int(config.audio_config.sample_rate * config.audio_config.min_duration)
        random_cropper = functools.partial(transforms.random_crop_signal, slice_length=desired_seq_len)
        center_cropper = functools.partial(transforms.center_crop_signal, slice_length=desired_seq_len)
    elif config.data.reader == "spec" or config.audio_config.features == "precomputed":
        # TODO: REFACTOR IF/ELSE block
        desired_seq_len = config.audio_config.get("timesteps", None)
        if desired_seq_len:
            random_cropper = functools.partial(transforms.random_crop_signal, slice_length=int(desired_seq_len))
            center_cropper = functools.partial(transforms.center_crop_signal, slice_length=int(desired_seq_len))
        else:
            random_cropper = None
            center_cropper = None
    else:
        desired_seq_len = int(config.audio_config.sample_rate * 1.)
        random_cropper = None
        center_cropper = None

    label_parser_func = functools.partial(transforms.label_parser, mode=config.model.type,
                                          num_classes=config.model.num_classes)

    parse_record_train = get_tfrecord_parser(config, cropper=random_cropper,
                                             label_parser_func=label_parser_func)
    parse_record_val = get_tfrecord_parser(config, cropper=center_cropper,
                                           label_parser_func=label_parser_func,
                                           is_val=True)
    fe = None
    if fe and not config.data.jax_transforms:
        batched_feature_extractor = functools.partial(transforms.map_torch_batched_feature_extractor,
                                                      feature_extractor=fe)
    else:
        batched_feature_extractor = None
    shuffle_buffer_multiplier = config.get("shuffle_buffer_multiplier", 10)
    train_dataset = dataset.get_dataset_v2(
        train_files, batch_size, 
        parse_example=parse_record_train, 
        num_classes=config.model.num_classes,
        compression=config.data.get("compression", "ZLIB"), 
        feature_extraction_func=batched_feature_extractor, 
        cacheable=config.data.cacheable, 
        shuffle=True, 
        num_shards=num_shards, 
        worker_index=worker_index,
        shuffle_buffer_multiplier=shuffle_buffer_multiplier
    )

    val_dataset = dataset.get_dataset_v2(
        val_files, batch_size, 
        parse_example=parse_record_val, 
        num_classes=config.model.num_classes,
        compression=config.data.get("compression", "ZLIB"),
        feature_extraction_func=batched_feature_extractor, 
        cacheable=config.data.cacheable,
        shuffle=False
    )

    if not config.data.jax_transforms:
        dtype_map_func = functools.partial(transforms.map_dtype, desired=input_dtype)
        train_dataset = train_dataset.map(dtype_map_func)
        val_dataset = val_dataset.map(dtype_map_func)
    return train_dataset, val_dataset
