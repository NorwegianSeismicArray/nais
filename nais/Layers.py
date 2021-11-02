import tensorflow as tf
from kapre import STFT, Magnitude

from tensorflow import keras
import tensorflow_addons as tfa

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

class ResidualConv1D(keras.layers.Layer):
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
                keras.layers.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same',
                                    activation='sigmoid'))
            self.tanh_layers.append(
                keras.layers.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same',
                                    activation='tanh'))
            self.conv_layers.append(keras.layers.Conv1D(self.filters, 1, padding='same'))

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


class ResidualConv1DTranspose(keras.layers.Layer):
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
            self.sigmoid_layers.append(keras.layers.Conv1DTranspose(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same', activation='sigmoid'))
            self.tanh_layers.append(keras.layers.Conv1DTranspose(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same', activation='tanh'))
            self.conv_layers.append(keras.layers.Conv1DTranspose(self.filters, 1, padding='same'))

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