
import tensorflow as tf 

class RandomCrop1D(tf.keras.layers.Layer):

    def __init__(self, crop=0.1, name='RandomCrop1D'):
        """
        Crop waveform data randomly.

        Args:
            crop (float) : proportion to crop. default 0.1.
            name (str) : Defaults to RandomCrop1D
        """
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