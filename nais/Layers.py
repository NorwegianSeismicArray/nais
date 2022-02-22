import tensorflow as tf
from kapre import STFT, Magnitude
import numpy as np

class StackedSpectrogram(tf.keras.layers.Layer):
    """
    Creates spectrograms for each channel and stacks them to grayscale.

    output_dim :: tuple
    n_fft :: int
    win_len :: int
    output_dim : tuple, resizing
    num_components : int, number of channels
    stack_method : str, add, mean, concat, None. 
            concat acts on frequency dimension. None concatenates last dim. 
    """
    def __init__(self, n_fft=2048, win_length=128, hop_length=32, output_dim=(64,64), num_components=3, stack_method=None, name='SpectrogramModel'):
        super(StackedSpectrogram, self).__init__(name=name)

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.output_dim = output_dim
        self.num_components = num_components
        self.stack_method = stack_method

    def get_config(self):
        return dict(n_fft = self.n_fft,
                win_length = self.win_length,
                hop_length = self.hop_length,
                output_dim = self.output_dim,
                num_components = self.num_components,
                stack_method = self.stack_method,
                name=self.name)

    def build(self,input_shape):
        inp = tf.keras.layers.Input(input_shape[1:])
        x = STFT(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)(inp)
        x = Magnitude()(x)
        x = tf.keras.layers.Lambda(tf.math.square)(x)
        x = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 1e-11, np.inf))(x)
        x = tf.keras.layers.Lambda(tf.math.log)(x)
        x = tf.keras.layers.Lambda(lambda x: tf.split(x, x.shape[-1]//self.num_components, axis=-1))(x)
        if self.stack_method=='add':
            x = tf.keras.layers.Add()(x)
        elif self.stack_method=='mean':
            x = tf.keras.layers.Average()(x)
        elif self.stack_method=='concat':
            x = tf.keras.layers.Concatenate(axis=1)(x)
        else:
            x = tf.keras.layers.Concatenate(axis=-1)(x)
            
        x = tf.keras.layers.Resizing(*self.output_dim)(x)
        self.model = tf.keras.Model(inp,x)
        
    def call(self, inputs):
        return self.model(inputs)

class ResidualConv1D(tf.keras.layers.Layer):
    def __init__(self, filters=32, kernel_size=3, stacked_layer=1):
        super(ResidualConv1D, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stacked_layer = stacked_layer

    def build(self, input_shape):
        self.sigmoid_layers = []
        self.tanh_layers = []
        self.conv_layers = []

        for dilation_rate in [2 ** i for i in range(self.stacked_layer)]:
            self.sigmoid_layers.append(
                tf.keras.layers.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same',
                                    activation='sigmoid'))
            self.tanh_layers.append(
                tf.keras.layers.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same',
                                    activation='tanh'))
            self.conv_layers.append(tf.keras.layers.Conv1D(self.filters, 1, padding='same'))

    def get_config(self):
        return dict(name=self.name,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    stacked_layer=self.stacked_layer)

    def call(self, inputs):
        residual_output = inputs
        x = inputs
        for sl, tl, cl in zip(self.sigmoid_layers, self.tanh_layers, self.conv_layers):
            sigmoid_x = sl(x)
            tanh_x = tl(x)

            x = tf.keras.layers.multiply([sigmoid_x, tanh_x])
            x = cl(x)
            residual_output = tf.keras.layers.add([residual_output, x])

        return residual_output


class ResidualConv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters=32, kernel_size=3, stacked_layer=1):
        super(ResidualConv1DTranspose, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stacked_layer = stacked_layer

    def build(self, input_shape):
        self.sigmoid_layers = []
        self.tanh_layers = []
        self.conv_layers = []

        for dilation_rate in [2 ** i for i in range(self.stacked_layer)]:
            self.sigmoid_layers.append(tf.keras.layers.Conv1DTranspose(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same', activation='sigmoid'))
            self.tanh_layers.append(tf.keras.layers.Conv1DTranspose(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same', activation='tanh'))
            self.conv_layers.append(tf.keras.layers.Conv1DTranspose(self.filters, 1, padding='same'))

    def get_config(self):
        return dict(name=self.name,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    stacked_layer=self.stacked_layer)

    def call(self, inputs):
        residual_output = inputs
        x = inputs
        for sl, tl, cl in zip(self.sigmoid_layers, self.tanh_layers, self.conv_layers):
            sigmoid_x = sl(x)
            tanh_x = tl(x)

            x = tf.keras.layers.multiply([sigmoid_x, tanh_x])
            x = cl(x)
            residual_output = tf.keras.layers.add([residual_output, x])

        return residual_output
    
class Resampling1D(tf.keras.layers.Layer):
    def __init__(self, length, interpolation="bilinear", **kwargs):
        super(Resampling1D, self).__init__(**kwargs)
        self.length = length
        self.interpolation = interpolation
        
    def build(self, input_shape):
        self.ls = tf.keras.layers.Resizing(self.length, input_shape[-1], interpolation=self.interpolation)

    def call(self,inputs):
        x = tf.expand_dims(inputs,axis=-1)
        x = self.ls(x)
        return tf.squeeze(x, axis=-1)

    def get_config(self):
        return dict(name=self.name, length=self.length)


class CosSimConv1D(tf.keras.layers.Layer):
    def __init__(self, units=32, kernel_size=3):
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


class MaxAbsPool1D(tf.keras.layers.Layer):
    def __init__(self, pool_size, pad_to_fit=False):
        super(MaxAbsPool1D, self).__init__()
        self.pad = pad_to_fit
        self.pool_size = pool_size

    def compute_output_shape(self, in_shape):
        if self.pad:
            return (in_shape[0],
                    tf.math.ceil(in_shape[1] / self.pool_size),
                    in_shape[2])
        return (in_shape[0],
                (in_shape[1] // self.pool_size),
                in_shape[2])

    def compute_padding(self, in_shape):
        mod_y = in_shape[1] % self.pool_size
        y1 = mod_y // 2
        y2 = mod_y - y1
        self.padding = ((0, 0), (y1, y2), (0, 0))

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_shape = self.compute_output_shape(self.in_shape)
        self.compute_padding(self.in_shape)

    @tf.function
    def stack(self, inputs):
        if self.pad:
            inputs = tf.pad(inputs, self.padding)
        batch_size = tf.shape(inputs)[0]
        max_height = (tf.shape(inputs)[1] // self.pool_size) * self.pool_size
        stack = tf.stack(
            [inputs[:, i:max_height:self.pool_size, :]
             for i in range(self.pool_size)],
            axis=-1)
        return stack

    @tf.function
    def call(self, inputs, training=None):
        stacked = self.stack(inputs)
        inds = tf.argmax(tf.abs(stacked), axis=-1, output_type=tf.int32)
        ks = tf.shape(stacked)
        idx = tf.stack([
            *tf.meshgrid(
                tf.range(0, ks[0]),
                tf.range(0, ks[1]),
                tf.range(0, ks[2]),
                indexing='ij'
            ), inds],
            axis=-1)

        x = tf.gather_nd(stacked, idx)
        x = tf.reshape(x, (-1, *self.out_shape[1:]))
        return x




