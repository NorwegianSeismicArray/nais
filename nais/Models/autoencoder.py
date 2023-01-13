

import tensorflow.keras.layers as tfl
import tensorflow as tf 

class ImageEncoder(tf.keras.Model):
    def __init__(self, depth=1):
        """2D encoder

        Args:
            depth (int, optional): numhber of layers. Defaults to 1.
        """
        super(ImageEncoder, self).__init__()
        self.model = tf.keras.Sequential()
        for d in range(depth):
            self.model.add(tfl.BatchNormalization())
            self.model.add(tfl.SpatialDropout2D(0.2))
            self.model.add(tfl.Conv2D(64, 7 * depth - 7 * d, strides=4, activation='relu', padding='same'))
        self.model.add(tfl.ActivityRegularization(l1=1e-3))

    def call(self, inputs):
        return self.model(inputs)

class ImageDecoder(tf.keras.Model):
    def __init__(self, depth=1, num_channels=3):
        """2D decoder

        Args:
            depth (int, optional): number of layers. Defaults to 1.
            num_channels (int, optional): number of output channels. Defaults to 3.
        """
        super(ImageDecoder, self).__init__()
        self.model = tf.keras.Sequential()
        for d in list(range(depth))[::-1]:
            self.model.add(tfl.BatchNormalization())
            self.model.add(tfl.SpatialDropout2D(0.2))
            self.model.add(
                tfl.Conv2DTranspose(64, 7 * depth - 7 * d, strides=4, activation='relu', padding='same'))
        self.model.add(tfl.Conv2D(num_channels, (3, 3), padding='same'))

    def call(self, inputs):
        return self.model(inputs)


class WaveEncoder(tf.keras.Model):
    def __init__(self, depth=1):
        """1D encoder

        Args:
            depth (int, optional): number of layers. Defaults to 1.
        """
        super(WaveEncoder, self).__init__()
        self.model = tf.keras.Sequential()
        for d in range(depth):
            self.model.add(tfl.BatchNormalization())
            self.model.add(tfl.SpatialDropout1D(0.2))
            self.model.add(tfl.Conv1D(64, 7 * depth - 7 * d, strides=4, activation='relu', padding='same'))
        self.model.add(tfl.ActivityRegularization(l1=1e-3))

    def call(self, inputs):
        return self.model(inputs)

    def predict(self, inputs, **kwargs):
        p = self.model.predict(inputs, **kwargs)
        return p.reshape((p.shape[0], -1))


class WaveDecoder(tf.keras.Model):
    def __init__(self, depth=1, 
                 num_channels=3):
        """1D decoder

        Args:
            depth (int, optional): number of layers. Defaults to 1.
            num_channels (int, optional): number of output channels. Defaults to 3.
        """
        super(WaveDecoder, self).__init__()
        self.model = tf.keras.Sequential()
        for d in list(range(depth))[::-1]:
            self.model.add(tfl.BatchNormalization())
            self.model.add(tfl.SpatialDropout1D(0.2))
            self.model.add(
                tfl.Conv1DTranspose(64, 7 * depth - 7 * d, strides=4, activation='relu', padding='same'))
        self.model.add(tfl.Conv1D(num_channels, 7, padding='same'))

    def call(self, inputs):
        return self.model(inputs)


class ImageAutoEncoder(tf.keras.Model):
    def __init__(self, 
                 num_outputs, 
                 depth=1, 
                 name='ImageAutoEncoder'):
        """2D autoencoder

        Args:
            num_outputs: (int): number of output channels. 
            depth (int, optional): number of encoder/decoder layers. Defaults to 1.
            name (str, optional): model name. Defaults to 'WaveAutoEncoder'.
        """
        super(ImageAutoEncoder, self).__init__(name=name)
        self.encoder = ImageEncoder(depth)
        self.decoder = ImageDecoder(depth, num_outputs)
        self.depth = depth

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

    def __str__(self):
        return self.name + f'-depth{self.depth}'


class WaveAutoEncoder(tf.keras.Model):
    def __init__(self, num_outputs, depth=1, name='WaveAutoEncoder'):
        """1D autoencoder

        Args:
            num_outputs: (int): number of output channels. 
            depth (int, optional): number of encoder/decoder layers. Defaults to 1.
            name (str, optional): model name. Defaults to 'WaveAutoEncoder'.
        """
        super(WaveAutoEncoder, self).__init__(name=name)
        self.encoder = WaveEncoder(depth)
        self.decoder = WaveDecoder(depth, num_outputs)
        self.depth = depth

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

    def __str__(self):
        return self.name + f'-depth{self.depth}'