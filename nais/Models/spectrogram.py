
import tensorflow as tf
import numpy as np
try:
    from kapre import STFT, Magnitude
except ImportError as e:
    print(e)
    print('kapre not installed')

import tensorflow as tf 
import tensorflow.keras.layers as tfl 

class CreateSpectrogramModel(tf.keras.Model):
    def __init__(self, 
                 n_fft=512, 
                 win_length=128,
                 hop_length=32, 
                 name='SpectrogramModel'):
        """
        Keras implementation of spectrograms.
        Stack infront of Conv2D model to create spectrograms on the fly.
        Note that, this is slower as it creates spectrograms per batch and not once.

        Args:
            n_fft (int, optional): number of FFTs. Defaults to 512.
            win_length (int, optional): window size in STFT. Defaults to 128.
            hop_length (int, optional): window stride in STFT. Defaults to 32.
            name (str, optional): model name. Defaults to 'SpectrogramModel'.
        """
        super(CreateSpectrogramModel, self).__init__(name=name)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.model = tf.keras.Sequential()

        self.model.add(STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        self.model.add(Magnitude())
        self.model.add(tfl.Lambda(tf.math.square))
        self.model.add(tfl.Lambda(lambda x: tf.clip_by_value(x, 1e-7, np.inf)))
        self.model.add(tfl.Lambda(tf.math.log))
        self.model.add(tfl.Resizing(256, 256))

    def call(self, inputs):
        return self.model(inputs)

    def __str__(self):
        return self.name + f'-n_fft-{self.n_fft}-win_length-{self.win_length}-hop_lenght-{self.hop_length}'
