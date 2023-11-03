
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

    def build(self, input_shape):
        # TODO: Verify this works
        self.length = int(input_shape[1] * (1 - self.crop))  # Use input_shape[1] to exclude batch size
        self.rc_layer = tf.keras.layers.RandomCrop(self.length, input_shape[2])  # input_shape[2] is the number of channels

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.rc_layer(x)
        x = tf.squeeze(x, axis=-1)
        return x