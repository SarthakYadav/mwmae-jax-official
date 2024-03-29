"""
Data transforms in tensorflow for the tf.data based datasets
Kept separate from core audax (since it's tensorflow based)

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import math
import tensorflow as tf


def db_to_linear(samples):
  return 10.0 ** (samples / 20.0)


def loudness_normalization(samples: tf.Tensor,
                           target_db: float = 15.0,
                           max_gain_db: float = 30.0):
  """Normalizes the loudness of the input signal."""
  std = tf.math.reduce_std(samples) + 1e-9
  gain = tf.minimum(db_to_linear(max_gain_db), db_to_linear(target_db) / std)
  return gain * samples


def maxabs_norm(x: tf.Tensor):
    """maxabs norm to [-1, 1]"""
    max_abs = tf.reduce_max(tf.abs(x), axis=0)
    if max_abs == 0.:
        max_abs = 1.
    return x / max_abs


def pad_waveform(waveform, seg_length=16000):
    padding = tf.maximum(seg_length - tf.shape(waveform)[0], 0)
    left_pad = padding // 2
    right_pad = padding - left_pad
    # print("in pad_waveform, {},{}".format(left_pad, right_pad))
    padded_waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]], mode="REFLECT")
    return padded_waveform


def random_crop_signal(audio, slice_length):
    data_length = tf.shape(audio, out_type=tf.dtypes.int64)[0]
    max_offset = data_length - slice_length
    if max_offset == 0:
        return audio
    random_offset = tf.random.uniform((), minval=0, maxval=max_offset, dtype=tf.dtypes.int64)
    slice_indices = tf.range(0, slice_length, dtype=tf.dtypes.int64)
    random_slice = tf.gather(audio, slice_indices + random_offset, axis=0)
    return random_slice


def center_crop_signal(audio, slice_length):
    data_length = tf.shape(audio, out_type=tf.dtypes.int64)[0]
    if data_length == slice_length:
        return audio
    center_offset = tf.maximum((data_length // 2) - (slice_length//2), 0)
    slice_indices = tf.range(0, slice_length, dtype=tf.dtypes.int64)
    return tf.gather(audio, slice_indices + center_offset, axis=0)


def label_parser(example, mode="multiclass", num_classes=527):
    label = tf.sparse.to_dense(example['label'])
    if mode == "multilabel":
        example['label'] = tf.reduce_sum(tf.one_hot(label, num_classes, on_value=1., axis=-1), axis=0)
    else:
        example['label'] = tf.one_hot(label[0], num_classes, on_value=1.)
    return example


def contrastive_labels(example):
    labels = tf.range(0, example['anchor'].shape[0])
    labels = tf.one_hot(labels, example['anchor'].shape[0], on_value=1.)
    example['label'] = labels
    return example


def map_torch_batched_feature_extractor(example, feature_extractor):
    example['audio'] = tf.numpy_function(feature_extractor, [example['audio']], Tout=tf.float32)
    return example


def map_dtype(example, desired=tf.float32):
    example['audio'] = tf.cast(example['audio'], desired)
    example['label'] = tf.cast(example['label'], desired)
    return example
