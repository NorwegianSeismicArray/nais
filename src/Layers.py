import tensorflow as tf
from kapre import STFT, Magnitude

class StackedSpectrogram(tf.keras.layers.Layer):
    """
    Creates spectrograms for each channel and stacks them to grayscale.

    output_dim :: tuple
    n_fft :: int
    win_len :: int

    """
    def __init__(self, n_fft=512, win_length=128, hop_length=32, output_dim=(64,64), name='SpectrogramModel'):
        super(StackedSpectrogram, self).__init__(name=name)

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.model = tf.keras.Sequential()

        self.model.add(STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        self.model.add(Magnitude())
        self.model.add(tf.keras.layers.Lambda(tf.math.square))
        self.model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 1e-7, np.inf)))
        self.model.add(tf.keras.layers.Concatenate(axis=-1))
        self.model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)))
        self.model.add(tf.keras.layers.Resizing(*output_dim))

    def call(self, inputs):
        return self.model(inputs)
