
import tensorflow.keras.layers as tfl 
import tensorflow as tf 
import numpy as np

try:
    from kapre import STFT, Magnitude
except ImportError as e:
    print(e)
    print('kapre not installed')

class StackedSpectrogram(tfl.Layer):
    """
    Creates spectrograms for each channel and stacks them to grayscale.

    Args:
        output_dim : tuple
        n_fft : int
        win_len : int
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
        inp = tfl.Input(input_shape[1:])
        x = STFT(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)(inp)
        x = Magnitude()(x)
        x = tfl.Lambda(tf.math.square)(x)
        x = tfl.Lambda(lambda x: tf.clip_by_value(x, 1e-11, np.inf))(x)
        x = tfl.Lambda(tf.math.log)(x)
        x = tfl.Lambda(lambda x: tf.split(x, x.shape[-1]//self.num_components, axis=-1))(x)
        if self.stack_method=='add':
            x = tfl.Add()(x)
        elif self.stack_method=='mean':
            x = tfl.Average()(x)
        elif self.stack_method=='concat':
            x = tfl.Concatenate(axis=1)(x)
        else:
            x = tfl.Concatenate(axis=-1)(x)
            
        x = tfl.Resizing(*self.output_dim)(x)
        self.model = tf.keras.Model(inp,x)
        
    def call(self, inputs):
        return self.model(inputs)