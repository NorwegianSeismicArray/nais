
import numpy as np
import tensorflow as tf

class RandomCrop1D(tf.keras.layers.Layer):
    """
    Crop waveform data randomly.

    crop :: float, proportion to crop. default 0.1.
    name :: str
    """
    def __init__(self, crop=0.1, name='RandomCrop1D'):
        super(RandomCrop1D, self).__init__(name=name)
        self.crop = crop

    def build(self, input_dim):
        _, x_size, y_size = input_dim
        self.length = x_size

    def call(self, inputs):
        x = tf.expand_dims(inputs,axis=-1)
        x = tf.keras.layers.RandomCrop(int(inputs.shape[1]*(1-self.crop)), inputs.shape[-1])(x)
        return x

class SpectrogramTimeAugment(tf.keras.layers.Layer):
    def __init__(self, prop=0.1, name='SpectrogramTimeAugment'):
        super(SpectrogramTimeAugment, self).__init__(name=name)
        self.prop = prop

    def call(self, inputs):
        _, length, _ = inputs.shape[1:]
        mask = tf.ones_like(inputs)
        start = tf.random.uniform(shape=(), minval=0, maxval=length * (1 - self.prop), dtype=tf.dtypes.int32)
        mask[:, :, start:start + self.prop * length, :] = 0.0
        return inputs * mask


class SpectrogramFreqAugment(tf.keras.layers.Layer):
    def __init__(self, prop=0.1, name='SpectrogramFreqAugment'):
        super(SpectrogramFreqAugment, self).__init__(name=name)
        self.prop = prop

    def call(self, inputs):
        height, _, _ = inputs.shape[1:]
        mask = tf.ones_like(inputs)
        start = tf.random.uniform(shape=(), minval=0, maxval=height * (1 - self.prop), dtype=tf.dtypes.int32)
        mask[:, start:start + self.prop * height, :, :] = 0.0
        return inputs * mask
