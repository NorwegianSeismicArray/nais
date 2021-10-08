"""
Author: Erik B. Myklebust, erik@norsar.no
2021
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from kapre import STFT, Magnitude


class ImageEncoder(keras.Model):
    def __init__(self, depth=1):
        super(ImageEncoder, self).__init__()
        self.model = keras.Sequential()
        for d in range(depth):
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.SpatialDropout2D(0.2))
            self.model.add(keras.layers.Conv2D(64, 7 * depth - 7 * d, strides=4, activation='relu', padding='same'))
        self.model.add(keras.layers.ActivityRegularization(l1=1e-3))

    def call(self, inputs):
        return self.model(inputs)

class ImageDecoder(keras.Model):
    def __init__(self, depth=1, num_channels=3):
        super(ImageDecoder, self).__init__()
        self.model = keras.Sequential()
        for d in list(range(depth))[::-1]:
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.SpatialDropout2D(0.2))
            self.model.add(
                keras.layers.Conv2DTranspose(64, 7 * depth - 7 * d, strides=4, activation='relu', padding='same'))
        self.model.add(keras.layers.Conv2D(num_channels, (3, 3), padding='same'))

    def call(self, inputs):
        return self.model(inputs)

class WaveEncoder(keras.Model):
    def __init__(self, depth=1):
        super(WaveEncoder, self).__init__()
        self.model = keras.Sequential()
        for d in range(depth):
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.SpatialDropout1D(0.2))
            self.model.add(keras.layers.Conv1D(64, 7 * depth - 7 * d, strides=4, activation='relu', padding='same'))
        self.model.add(keras.layers.ActivityRegularization(l1=1e-3))

    def call(self, inputs):
        return self.model(inputs)

    def predict(self, inputs, **kwargs):
        p = self.model.predict(inputs, **kwargs)
        return p.reshape((p.shape[0], -1))


class WaveDecoder(keras.Model):
    def __init__(self, depth=1, num_channels=3):
        super(WaveDecoder, self).__init__()
        self.model = keras.Sequential()
        for d in list(range(depth))[::-1]:
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.SpatialDropout1D(0.2))
            self.model.add(
                keras.layers.Conv1DTranspose(64, 7 * depth - 7 * d, strides=4, activation='relu', padding='same'))
        self.model.add(keras.layers.Conv1D(num_channels, 7, padding='same'))

    def call(self, inputs):
        return self.model(inputs)


class ImageAutoEncoder(keras.Model):
    def __init__(self, depth=1, name='ImageAutoEncoder'):
        super(ImageAutoEncoder, self).__init__(name=name)
        self.encoder = ImageEncoder(depth)
        self.decoder = ImageDecoder(depth)
        self.depth = depth

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

    def __str__(self):
        return self.name + f'-depth{self.depth}'


class WaveAutoEncoder(keras.Model):
    def __init__(self, depth=1, name='WaveAutoEncoder'):
        super(WaveAutoEncoder, self).__init__(name=name)
        self.encoder = WaveEncoder(depth)
        self.decoder = WaveDecoder(depth)
        self.depth = depth

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

    def __str__(self):
        return self.name + f'-depth{self.depth}'


class CreateSpectrogramModel(keras.Model):
    def __init__(self, n_fft=512, win_length=128, hop_length=32, name='SpectrogramModel'):
        super(CreateSpectrogramModel, self).__init__(name=name)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.model = keras.Sequential()

        self.model.add(STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        self.model.add(Magnitude())
        self.model.add(keras.layers.Lambda(tf.math.square))
        self.model.add(keras.layers.Lambda(lambda x: tf.clip_by_value(x, 1e-7, np.inf)))
        self.model.add(keras.layers.Resizing(256, 256))

    def call(self, inputs):
        return self.model(inputs)

    def __str__(self):
        return self.name + f'-n_fft-{self.n_fft}-win_length-{self.win_length}-hop_lenght-{self.hop_length}'


class AlexNet(keras.Model):
    """
    https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
    """

    def __init__(self, kernel_sizes=None, num_classes=None, pooling='max', name='AlexNet'):
        super(AlexNet, self).__init__(name=name)
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        assert len(kernel_sizes) == 5
        assert pooling in [None, 'max', 'avg']

        if pooling == 'max':
            pooling_layer = tf.keras.layers.MaxPooling2D
        elif pooling == 'avg':
            pooling_layer = tf.keras.layers.AveragePooling2D
        else:
            pooling_layer = lambda **kwargs: tf.keras.layers.Activation('linear')

        self.layers = [
            tf.keras.layers.Conv2D(filters=96, kernel_size=kernel_sizes[0], strides=(4, 4), activation='relu',
                                   padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=kernel_sizes[1], strides=(1, 1), activation='relu',
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            pooling_layer(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=384, kernel_size=kernel_sizes[2], strides=(1, 1), activation='relu',
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=384, kernel_size=kernel_sizes[3], strides=(1, 1), activation='relu',
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=kernel_sizes[4], strides=(1, 1), activation='relu',
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            pooling_layer(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
        ]

        if num_classes is not None:
            assert type(num_classes) == int
            self.layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class WaveAlexNet(keras.Model):
    """
    Save as AlexNet but with 1D convolutions.
    """

    def __init__(self, kernel_sizes=None, num_classes=None, pooling='max', name='WaveAlexNet'):
        super(WaveAlexNet, self).__init__(name=name)
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        assert len(kernel_sizes) == 5
        assert pooling in [None, 'max', 'avg']

        if pooling == 'max':
            pooling_layer = tf.keras.layers.MaxPooling1D
        elif pooling == 'avg':
            pooling_layer = tf.keras.layers.AveragePooling1D
        else:
            pooling_layer = lambda **kwargs: tf.keras.layers.Activation('linear')

        self.layers = [
            tf.keras.layers.Conv1D(filters=96, kernel_size=kernel_sizes[0], strides=1, activation='relu',
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=3, strides=2),
            tf.keras.layers.Conv1D(filters=256, kernel_size=kernel_sizes[1], strides=1, activation='relu',
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tf.keras.layers.Conv1D(filters=384, kernel_size=kernel_sizes[2], strides=1, activation='relu',
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(filters=384, kernel_size=kernel_sizes[3], strides=1, activation='relu',
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(filters=256, kernel_size=kernel_sizes[4], strides=1, activation='relu',
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            pooling_layer(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
        ]

        if num_classes is not None:
            assert type(num_classes) == int
            self.layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class PhaseNet(keras.Model):
    """
    Adapted from https://keras.io/examples/vision/oxford_pets_image_segmentation/

    args

    num_classes : int
        Number of output classes, eg. P and S wave picking num_classes=2.

    """
    def __init__(self, num_classes=2, filters=None, name='PhaseNet'):
        super(PhaseNet, self).__init__(name=name)
        self.num_classes = num_classes

        if filters is None:
            self.filters = [4, 8, 16, 32]
        else:
            self.filters = filters

    def build(self, input_shape):
        inputs = keras.Input(shape=input_shape[1:])

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = keras.layers.Conv1D(self.filters[0], 7, strides=2, padding="same")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in self.filters[1:]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv1D(filters, 7, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv1D(filters, 7, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.MaxPooling1D(4, strides=2, padding="same")(x)

            # Project residual
            residual = keras.layers.Conv1D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in self.filters[::-1]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv1DTranspose(filters, 7, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv1DTranspose(filters, 7, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.UpSampling1D(2)(x)

            # Project residual
            residual = keras.layers.UpSampling1D(2)(previous_block_activation)
            residual = keras.layers.Conv1D(filters, 1, padding="same")(residual)
            x = keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = keras.layers.Conv1D(self.num_classes, 3, activation="softmax", padding="same")(x)

        # Define the model
        self.model = keras.Model(inputs, outputs)

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)