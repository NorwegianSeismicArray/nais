import tensorflow.keras.layers as tfl 
import tensorflow as tf 

class Resampling1D(tfl.Layer):
    def __init__(self, length, interpolation="bilinear", **kwargs):
        """Wrapper around 2D resampling layer to apply to 1D input.

        Args:
            length (int): length of output
            interpolation (str): interpolation type
        """
        super(Resampling1D, self).__init__(**kwargs)
        self.length = length
        self.interpolation = interpolation
        
    def build(self, input_shape):
        self.ls = tfl.Resizing(self.length, input_shape[-1], interpolation=self.interpolation)

    def call(self,inputs):
        x = tf.expand_dims(inputs,axis=-1)
        x = self.ls(x)
        return tf.squeeze(x, axis=-1)

    def get_config(self):
        return dict(name=self.name, length=self.length)

