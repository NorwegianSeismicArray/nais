
import tensorflow as tf
import numpy as np

class RandomCrop1D(tf.keras.layers.Layer):
    """
    Crop waveform data randomly.

    crop :: float, proportion to crop. default 0.1.
    name :: str
    """

    def __init__(self, crop=0.1, name='RandomCrop1D'):
        super(RandomCrop1D, self).__init__(name=name)
        self.crop = crop

    def get_config(self):
        return dict(crop=self.crop, name=self.name)

    def build(self, input_dim):
        _, x_size, y_size = input_dim
        self.length = int(x_size * (1 - self.crop))
        self.channels = y_size
        self.rc_layer = tf.keras.layers.RandomCrop(self.length, self.channels)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.rc_layer(x)
        x = tf.squeeze(x, axis=-1)
        return x

class SpectrogramTimeAugment(tf.keras.layers.Layer):
    def __init__(self, prop=0.1, name='SpectrogramTimeAugment'):
        super(SpectrogramTimeAugment, self).__init__(name=name)
        self.prop = prop

    def get_config(self):
        return dict(prop=self.prop, name=self.name)

    def call(self, inputs):
        length, _, _ = inputs.shape[1:]
        mask = np.ones(inputs.shape[1:])
        start = np.random.randint(0, int(length * (1 - self.prop)))
        mask[start:start + int(self.prop * length), :, :] = 0.0
        return inputs * np.expand_dims(mask, axis=0)

class SpectrogramFreqAugment(tf.keras.layers.Layer):
    def __init__(self, prop=0.1, name='SpectrogramFreqAugment'):
        super(SpectrogramFreqAugment, self).__init__(name=name)
        self.prop = prop

    def get_config(self):
        return dict(prop=self.prop, name=self.name)

    def call(self, inputs):
        _, height, _ = inputs.shape[1:]
        mask = np.ones(inputs.shape[1:])
        start = np.random.randint(0, int(height * (1 - self.prop)))
        mask[:, start:start + int(self.prop * height), :] = 0.0
        return inputs * np.expand_dims(mask, axis=0)
