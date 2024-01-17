"""
Audio dataset and parsing utilities for audax

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import os
from absl import logging
import tensorflow as tf
import tensorflow_io as tfio
from typing import Callable, Optional
from functools import partial
from .transforms import pad_waveform


def parse_tfrecord_fn_v2(example,
                         label_parser=None,
                         cropper=None,
                         seg_length=16000):
    feature_description = {
        "audio": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.VarLenFeature(tf.int64),
        "duration": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    a = tf.io.parse_tensor(example['audio'], tf.float32)
    a.set_shape([None])
    a = pad_waveform(a, seg_length=seg_length)
    if cropper:
        a = tf.numpy_function(cropper, [a], tf.float32)
    example['audio'] = a
    if label_parser:
        example = label_parser(example)
    return example


def parse_tfrecord_specgram(example,
                            cropper=None,
                            normalize_spec=True,
                            label_parser=None,
                            flip_ft_axis=False):
    feature_description = {
        "audio": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    a = tf.io.parse_tensor(example['audio'], tf.float32)
    if normalize_spec:
        a = (a - tf.reduce_mean(a)) / (tf.math.reduce_std(a) + 1e-8)
    if cropper:
        a = cropper(a)
    if flip_ft_axis:
        a = tf.transpose(a, perm=(1, 0))
    example['audio'] = a
    if label_parser:
        example = label_parser(example)
    return example


def get_dataset_v2(filenames,
                batch_size,
                parse_example,
                num_classes,
                file_ext="tfrec",
                compression="ZLIB",
                feature_extraction_func=None,
                cacheable=False,
                is_contrastive=False,
                shuffle=False,
                num_shards=None,
                worker_index=None,
                shuffle_buffer_multiplier=10
                ):
    options = tf.data.Options()
    options.autotune.enabled = True
    options.threading.private_threadpool_size = 96  # 0=automatically determined
    options.deterministic = False
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.with_options(options)
    # shuffle filenames every epoch

    if num_shards:
        dataset = dataset.shuffle(len(filenames), seed=0, reshuffle_each_iteration=False)
        logging.info("SHARDING dataset >>>")
        dataset = dataset.shard(num_shards, worker_index)
    else:
        dataset = dataset.shuffle(len(filenames), seed=0, reshuffle_each_iteration=True)
    # dataset = tf.data.TFRecordDataset(filenames, compression_type=compression,
    #                                       num_parallel_reads=tf.data.AUTOTUNE)
    # dataset = dataset.with_options(options)
    # dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    if cacheable:
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=compression,
                                            num_parallel_reads=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE),
            block_length=2,
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # gives better data throughput for AudioSet, which can't really be cached anyway
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=compression,
                                              num_parallel_reads=tf.data.AUTOTUNE).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE),
            block_length=2,
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(batch_size*shuffle_buffer_multiplier, seed=0, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
    if feature_extraction_func:
        dataset = dataset.map(feature_extraction_func,
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
