import tensorflow.keras.layers as tfl 
import tensorflow as tf 

class CosSimConv1D(tfl.Layer):
    
    def __init__(self, units=32, kernel_size=3):
        """
        Cosine similarity convolution 
        Args:
            units (int): number of filters
            kernel_size (int): size of filters
        """
        super(CosSimConv1D, self).__init__()
        self.units = units
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.in_shape = input_shape

        self.flat_size = self.in_shape[1]
        self.channels = self.in_shape[2]

        self.w = self.add_weight(
            shape=(1, self.channels * self.kernel_size, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True)

        self.p = self.add_weight(
            shape=(self.units,), initializer='ones', trainable=True)

        self.q = self.add_weight(
            shape=(1,), initializer='zeros', trainable=True)

    def l2_normal(self, x, axis=None, epsilon=1e-12):
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_inv_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
        return x_inv_norm

    def stack3(self, x):
        x = tf.stack(
            [
                tf.pad(x[:, :-1, :], tf.constant([[0, 0], [1, 0], [0, 0]])),
                x,
                tf.pad(x[:, 1:, :], tf.constant([[0, 0], [0, 1], [0, 0]])),
            ], axis=2)
        return x

    def call(self, inputs, training=None):
        x = self.stack3(inputs)
        x = tf.reshape(x, (-1, self.flat_size, self.channels * self.kernel_size))
        q = tf.square(self.q)
        x_norm = self.l2_normal(x, axis=2) + q
        w_norm = self.l2_normal(self.w, axis=1) + q
        sign = tf.sign(tf.matmul(x, self.w))
        x = tf.matmul(x / x_norm, self.w / w_norm)
        x = tf.abs(x) + 1e-12
        x = tf.pow(x, tf.square(self.p))
        x = sign * x + self.b
        x = tf.reshape(x, (-1, self.in_shape[1], self.units))
        return x
