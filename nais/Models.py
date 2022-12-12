"""
Author: Erik B. Myklebust, erik@norsar.no
2021
"""

import tensorflow as tf
import numpy as np
try:
    from kapre import STFT, Magnitude
except ImportError as e:
    print(e)
    print('kapre not installed')

import tensorflow.keras.layers as tfl
from nais.Layers import ResidualConv1D, ResnetBlock1D, SeqSelfAttention, FeedForward, Scattering, ScatteringV2
from nais.Mixture import GMM
from nais.Cluster import KMeans
import tensorflow.keras.backend as K

class ImageEncoder(tf.keras.Model):
    def __init__(self, depth=1):
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
    def __init__(self, depth=1, num_channels=3):
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
    """
    Autoencoder which encodes images.

    depth : int
        number of convolutional layers
    """
    def __init__(self, depth=1, name='ImageAutoEncoder'):
        super(ImageAutoEncoder, self).__init__(name=name)
        self.encoder = ImageEncoder(depth)
        self.decoder = ImageDecoder(depth)
        self.depth = depth

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

    def __str__(self):
        return self.name + f'-depth{self.depth}'


class WaveAutoEncoder(tf.keras.Model):
    """
    Autoencoder which encodes waveforms.

    depth : int
        number of convolutional layers
    """

    def __init__(self, depth=1, name='WaveAutoEncoder'):
        super(WaveAutoEncoder, self).__init__(name=name)
        self.encoder = WaveEncoder(depth)
        self.decoder = WaveDecoder(depth)
        self.depth = depth

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

    def __str__(self):
        return self.name + f'-depth{self.depth}'


class CreateSpectrogramModel(tf.keras.Model):
    """
    Keras implementation of spectrograms.
    Stack infront of Conv2D model to create spectrograms on the fly.
    Note that, this is slower as it creates spectrograms per batch and not once.
    """
    def __init__(self, n_fft=512, win_length=128, hop_length=32, name='SpectrogramModel'):
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


class AlexNet2D(tf.keras.Model):
    """
    https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98

    kernel_size : list of length 5
        kernel sizes to use in model
    num_outputs : int or None
        number of outputs of final dense layer. Leave as None to exclude top.
    output_type : str
        type of output, binary, multiclass, multilabel, regression
    pooling : str
         pooling type, max or avg, other will use no pooling
    """

    def __init__(self, kernel_sizes=None, num_outputs=None, output_type='binary', pooling='max', name='AlexNet2D'):
        super(AlexNet2D, self).__init__(name=name)
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        assert len(kernel_sizes) == 5
        assert pooling in [None, 'max', 'avg']

        if pooling == 'max':
            pooling_layer = tfl.MaxPooling2D
        elif pooling == 'avg':
            pooling_layer = tfl.AveragePooling2D
        else:
            pooling_layer = lambda **kwargs: tfl.Activation('linear')

        self.ls = [
            tfl.Conv2D(filters=96, kernel_size=kernel_sizes[0], strides=(4, 4), activation='relu',
                                   padding='same'),
            tfl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tfl.Conv2D(filters=256, kernel_size=kernel_sizes[1], strides=(1, 1), activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=(3, 3), strides=(2, 2)),
            tfl.Conv2D(filters=384, kernel_size=kernel_sizes[2], strides=(1, 1), activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=384, kernel_size=kernel_sizes[3], strides=(1, 1), activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=256, kernel_size=kernel_sizes[4], strides=(1, 1), activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=(2, 2)),
            tfl.Flatten(),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
        ]

        if num_outputs is not None:
            if output_type == 'binary':
                assert num_outputs == 1
                act = 'sigmoid'
            elif output_type == 'multiclass':
                assert num_outputs > 1
                act = 'softmax'
            elif output_type == 'multilabel':
                assert num_outputs > 1
                act = 'sigmoid'
            else:
                act = 'linear'

            self.ls.append(tfl.Dense(num_outputs, activation=act))

    def call(self, inputs):
        x = inputs
        for layer in self.ls:
            x = layer(x)
        return x


class AlexNet1D(tf.keras.Model):
    """
    Same as AlexNet but with 1D convolutions.

    kernel_size : list of length 5
        kernel sizes to use in model
    num_outputs : int or None
        number of outputs of final dense layer. Leave as None to exclude top.
    output_type : str
        type of output, binary, multiclass, multilabel, regression
    pooling : str
         pooling type, max or avg, other will use no pooling
    """

    def __init__(self, kernel_sizes=None, filters=None, num_outputs=None, output_type='binary', pooling='max', name='AlexNet1D'):
        super(AlexNet1D, self).__init__(name=name)
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        if filters is None:
          	filters = [96, 256, 384, 384, 256]
        assert len(kernel_sizes) == 5
        assert pooling in [None, 'none', 'max', 'avg']

        if pooling == 'max':
            pooling_layer = tfl.MaxPooling1D
        elif pooling == 'avg':
            pooling_layer = tfl.AveragePooling1D
        else:
            pooling_layer = lambda **kwargs: tfl.Activation('linear')

        self.ls = [
            tfl.Conv1D(filters=filters[0], kernel_size=kernel_sizes[0], strides=4, activation='relu',
                                   padding='same'),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Conv1D(filters=filters[1], kernel_size=kernel_sizes[1], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Conv1D(filters=filters[2], kernel_size=kernel_sizes[2], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv1D(filters=filters[3], kernel_size=kernel_sizes[3], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv1D(filters=filters[4], kernel_size=kernel_sizes[4], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Flatten(),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
        ]

        if num_outputs is not None:
            if output_type == 'binary':
                assert num_outputs == 1
                act = 'sigmoid'
            elif output_type == 'multiclass':
                assert num_outputs > 1
                act = 'softmax'
            elif output_type == 'multilabel':
                assert num_outputs > 1
                act = 'sigmoid'
            else:
                act = 'linear'

            self.ls.append(tfl.Dense(num_outputs, activation=act))

    def call(self, inputs):
        x = inputs
        for layer in self.ls:
            x = layer(x)
        return x

class PhaseNet(tf.keras.Model):
    """
    Adapted from https://keras.io/examples/vision/oxford_pets_image_segmentation/

    args

    num_classes : int
        Number of output classes, eg. P and S wave picking num_classes=2.

    """

    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 initializer='glorot_normal',
                 name='PhaseNet'):
        super(PhaseNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation

        if filters is None:
            self.filters = [4, 8, 16, 32]
        else:
            self.filters = filters

        if kernelsizes is None:
            self.kernelsizes = [7, 7, 7, 7]
        else:
            self.kernelsizes = kernelsizes

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=2,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation("relu")(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, filters in enumerate(self.filters[1:]):
            x = tfl.Activation("relu")(x)
            x = tfl.Conv1D(filters, self.kernelsizes[i+1], padding="same",
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer,
                           )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation("relu")(x)
            x = tfl.Dropout(self.dropout_rate)(x)

            x = tfl.Conv1D(filters, self.kernelsizes[i+1], padding="same",
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer,
                           )(x)
            x = tfl.BatchNormalization()(x)

            x = tfl.MaxPooling1D(4, strides=2, padding="same")(x)

            # Project residual
            residual = tfl.Conv1D(filters, 1, strides=2, padding="same", kernel_initializer=self.initializer,
                                  )(
                previous_block_activation
            )
            x = tfl.concatenate([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###

        for i, filters in enumerate(self.filters[::-1]):
            x = tfl.Activation("relu")(x)
            x = tfl.Conv1DTranspose(filters, self.kernelsizes[::-1][i], padding="same",
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_initializer=self.initializer,
                                    )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation("relu")(x)
            x = tfl.Dropout(self.dropout_rate)(x)

            x = tfl.Conv1DTranspose(filters, self.kernelsizes[::-1][i], padding="same",
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_initializer=self.initializer,
                                    )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.UpSampling1D(2)(x)

            # Project residual
            residual = tfl.UpSampling1D(2)(previous_block_activation)
            residual = tfl.Conv1D(filters, 1, padding="same",
                                  kernel_regularizer=self.kernel_regularizer,
                                  kernel_initializer=self.initializer,
                                  )(residual)
            x = tfl.concatenate([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        to_crop = x.shape[1] - input_shape[1]
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        x = tfl.Cropping1D((of_start, of_end))(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            outputs = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same",
                           activation=self.output_activation)(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

class UTime(tf.keras.Model):
    def __init__(self,
                 phasenet_filters=None,
                 output_filters=None,
                 output_activation='softmax',
                 num_classes=2,
                 pool_sizes=[4],
                 pool_strides=[2],
                 dropout_rate=0.2,
                 pool_type='avg',
                 name='UTime'):
        super(UTime, self).__init__(name=name)
        self.phasenet = PhaseNet(filters=phasenet_filters, num_classes=None)
        if output_filters is None:
            self.filters = [4]
        else:
            self.filters = output_filters

        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.pool_type = pool_type
        self.output_activation = output_activation
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.phasenet.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        x = self.phasenet(inputs)
        for i, psize, pstride in enumerate(zip(self.pool_sizes, self.pool_strides)):
            if self.pool_type == 'avg':
                x = tfl.AveragePooling1D(psize, strides=pstride, padding='same')(x)
            elif self.pool_type == 'max':
                x = tfl.MaxPooling1D(psize, strides=pstride, padding='same')(x)
            else:
                raise NotImplementedError(f'pool_type={self.pool_type} is not supported.')
            if i < len(self.pool_size) - 1: #skip when on last pool layer.
                x = tfl.Conv1D(self.filters[i], activation=None, padding='same')(x)
                x = tfl.BatchNormalization()(x)
                x = tfl.Activation('relu')(x)
                x = tfl.Dropout(self.dropout_rate)(x)

        outputs = tfl.Conv1D(self.num_classes, activation=self.output_activation, padding='same')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

class PhaseNetDist(tf.keras.Model):
    def __init__(self,
                 dist_filters=None,
                 output_activation=None,
                 num_outputs=1,
                 dropout_rate=0.2,
                 pool_type='max',
                 kernel_regularizer=None,
                 kernel_initializer='glorot_normal',
                 phasenet_num_classes=2,
                 phasenet_filters=None,
                 phasenet_output_activation='linear',
                 phasenet_kernel_regularizer=None,
                 phasenet_dropout_rate=0.2,
                 phasenet_initializer='glorot_normal',
                 name='PhaseNetDist'):
        super(PhaseNetDist, self).__init__(name=name)
        self.phasenet = PhaseNet(filters=phasenet_filters,
                                 num_classes=phasenet_num_classes,
                                 output_activation=phasenet_output_activation,
                                 dropout_rate=phasenet_dropout_rate,
                                 kernel_regularizer=phasenet_kernel_regularizer,
                                 initializer=phasenet_initializer)
        if dist_filters is None:
            self.filters = [8]
        else:
            self.filters = dist_filters
        self.pool_type = pool_type
        self.output_activation = output_activation
        self.num_outputs = num_outputs
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.phasenet.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        phasenet_output = self.phasenet(inputs)
        x = self.phasenet.encoder(inputs)

        for f in self.filters:
            x = tfl.Conv1D(f,
                           activation=None,
                           padding='same',
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer)(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation('relu')(x)
            x = tfl.Dropout(self.dropout_rate)(x)

        if self.pool_type == 'avg':
            x = tfl.GlobalAveragePooling1D()(x)
        elif self.pool_type == 'max':
            x = tfl.GlobalMaxPooling1D()(x)
        else:
            raise NotImplementedError(f'pool_type={self.pool_type} is not supported.')

        distance_output = tfl.Dense(self.num_outputs, activation=self.output_activation)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=[phasenet_output, distance_output])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

class WaveNet(tf.keras.Model):
    """
    https://deepmind.com/blog/article/wavenet-generative-model-raw-audio
    """

    def __init__(self, num_outputs=None, kernel_size=3, output_type='binary', pooling=None, filters=None, stacked_layers=None, name='WaveNet'):
        super(WaveNet, self).__init__(name=name)

        if filters is None:
            self.filters = 16
        else:
            self.filters = filters

        if stacked_layers is None:
            self.stacked_layers = [12,8,4,1]
        else:
            self.stacked_layers = stacked_layers

        self.ls = []
        for i,sl in enumerate(self.stacked_layers):
            self.ls.append(tfl.Conv1D(self.filters*(i+1), 1, padding='same'))
            self.ls.append(ResidualConv1D(self.filters*(i+1), kernel_size, sl))
		
        if pooling is None:
            self.ls.append(tfl.Flatten())
        elif pooling == 'avg':
            self.ls.append(tfl.GlobalAveragePooling1D())
        elif pooling == 'max':
            self.ls.append(tfl.GlobalMaxPooling1D())
        else:
            raise NotImplementedError(pooling + 'no implemented')
        
        if num_outputs is not None:
            if output_type == 'binary':
                assert num_outputs == 1
                act = 'sigmoid'
            elif output_type == 'multiclass':
                assert num_outputs > 1
                act = 'softmax'
            elif output_type == 'multilabel':
                assert num_outputs > 1
                act = 'sigmoid'
            else:
                act = 'linear'

            self.ls.append(tfl.Dense(num_outputs, activation=act))

    def call(self, inputs):
        x = inputs
        for layer in self.ls:
            x = layer(x)
        return x


class EarthQuakeTransformer(tf.keras.Model):
    """
    https://www.nature.com/articles/s41467-020-17591-w

    Example
    import numpy as np
    test = np.random.random(size=(16,1024,3))
    d = np.random.randint(2, size=(16,1024,1))
    p = np.random.randint(2, size=(16,1024,1))
    s = np.random.randint(2, size=(16,1024,1))

    model = EarthQuakeTransformer(input_dim=test.shape[1:])
    model.compile(optimizer='adam', loss=['binary_crossentropy',
                                          'binary_crossentropy',
                                          'binary_crossentropy'])

    model.fit(test, (d,p,s))

    """

    def __init__(self,
                 input_dim,
                 filters=None,
                 kernelsizes=None,
                 resfilters=None,
                 reskernelsizes=None,
                 lstmfilters=None,
                 attention_width=3,
                 dropout=0.0,
                 transformer_sizes=[64, 64],
                 kernel_regularizer=None,
                 classify=True,
                 att_type='additive',
                 name='EarthQuakeTransformer'):
        super(EarthQuakeTransformer, self).__init__(name=name)

        if filters is None:
            filters = [8, 16, 16, 32, 32, 64, 64]
        if kernelsizes is None:
            kernelsizes = [11, 9, 7, 7, 5, 5, 3]
        invfilters = filters[::-1]
        invkernelsizes = kernelsizes[::-1]
        if resfilters is None:
            resfilters = [64, 64, 64, 64, 64]
        if reskernelsizes is None:
            reskernelsizes = [3, 3, 3, 2, 2]
        if lstmfilters is None:
            lstmfilters = [16, 16]

        try:
            assert resfilters[0] == filters[-1]
        except AssertionError:
            print('Filters missmatch.')
            filters = resfilters[0]

        def conv_block(f,kz):
            return tf.keras.Sequential([tfl.Conv1D(f, kz, padding='same', kernel_regularizer=kernel_regularizer),
                                        tfl.BatchNormalization(),
                                        tfl.Activation('relu'),
                                        tfl.Dropout(dropout),
                                        tfl.MaxPooling1D(2, padding='same')])

        def block_BiLSTM(f, x):
            'Returns LSTM residual block'
            x = tfl.Bidirectional(tfl.LSTM(f, return_sequences=True))(x)
            x = tfl.Conv1D(f, 1, padding='same', kernel_regularizer=kernel_regularizer)(x)
            x = tfl.BatchNormalization()(x)
            return x

        def block_transformer(f, width, x):
            att, w = SeqSelfAttention(return_attention=True,
                                      attention_width=width,
                                      attention_type=att_type)(x)
            att = tfl.Add()([x, att])
            norm = tfl.LayerNormalization()(att)
            ff = tf.keras.Sequential([tfl.Dense(f, activation='relu', kernel_regularizer=kernel_regularizer),
                                      tfl.Dropout(dropout),
                                      tfl.Dense(norm.shape[2]),
                                      ])(norm)
            ff_add = tfl.Add()([norm, ff])
            norm_out = tfl.LayerNormalization()(ff_add)
            return norm_out, w

        def _encoder():
            inp = tfl.Input(input_dim)
            def encode(x):
                for f, kz in zip(filters, kernelsizes):
                    x = conv_block(f, kz)(x)
                for f, kz in zip(resfilters, reskernelsizes):
                    x = ResnetBlock1D(f, kz, dropout=dropout, kernel_regularizer=kernel_regularizer)(x)
                for f in lstmfilters:
                    x = block_BiLSTM(f, x)
                x = tfl.LSTM(64, return_sequences=True, kernel_regularizer=kernel_regularizer)(x)
                for ts in transformer_sizes:
                    x, w0 = block_transformer(ts, None, x)
                return x
            return tf.keras.Model(inp, encode(inp))

        def inv_conv_block(f,kz):
            return tf.keras.Sequential([tfl.UpSampling1D(2),
                                        tfl.Conv1D(f, kz, padding='same', kernel_regularizer=kernel_regularizer),
                                        tfl.BatchNormalization(),
                                        tfl.Activation('relu'),
                                        tfl.Dropout(dropout)])

        def _decoder(input_shape, attention=False, activation='sigmoid', output_name=None):
            inp = tfl.Input(input_shape)
            x = inp
            if attention:
                x = tfl.LSTM(filters[1], return_sequences=True, kernel_regularizer=kernel_regularizer)(x)
                x, w = SeqSelfAttention(return_attention=True,
                                        attention_width=attention_width,
                                        attention_type=att_type)(x)

            x = tf.keras.Sequential([inv_conv_block(f, kz) for f, kz in zip(invfilters, invkernelsizes)])(x)
            to_crop = x.shape[1] - input_dim[0]
            of_start, of_end = to_crop//2, to_crop//2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
            if not (activation is None):
                x = tfl.Conv1D(1, 3, padding='same', activation=activation, name=output_name)(x)
            return tf.keras.Model(inp, x)

        self.feature_extractor = _encoder()
        encoded_dim = self.feature_extractor.layers[-1].output.shape[1:]
        self.detector = _decoder(encoded_dim, attention=False, activation='sigmoid' if classify else None, output_name='detection')
        self.p_picker = _decoder(encoded_dim, attention=True, activation='sigmoid' if classify else None, output_name='p_phase')
        self.s_picker = _decoder(encoded_dim, attention=True, activation='sigmoid' if classify else None, output_name='s_phase')

    @property
    def num_parameters(self):
        s = 0
        for m in [self.feature_extractor, self.detector, self.s_picker, self.p_picker]:
            s += sum([np.prod(K.get_value(w).shape) for w in m.trainable_weights])
        return s

    def call(self, inputs):
        encoded = self.feature_extractor(inputs)
        d = self.detector(encoded)
        p = self.p_picker(encoded)
        s = self.s_picker(encoded)
        return d, p, s


class EarthQuakeTransformerMetadata(EarthQuakeTransformer):
    def __init__(self, num_outputs, eqt_kw):
        super(EarthQuakeTransformerMetadata, self).__init__(**eqt_kw)
        self.metadata_model = tf.keras.Sequential([tfl.Flatten(),
                                                   tfl.Dense(128, activation='relu'),
                                                   tfl.Dense(num_outputs)])

    def call(self, inputs):
        encoded = self.feature_extractor(inputs)
        d = self.detector(encoded)
        p = self.p_picker(encoded)
        s = self.s_picker(encoded)
        m = self.metadata_model(encoded)
        return d, p, s, m

class PhaseNetMetadata(PhaseNet):
    def __init__(self, num_outputs, ph_kw):
        super(PhaseNetMetadata, self).__init__(**ph_kw)
        self.metadata_model = tf.keras.Sequential([tfl.Flatten(),
                                                   tfl.Dense(128, activation='relu'),
                                                   tfl.Dense(num_outputs)])
    def call(self, inputs):
        p = self.model(inputs)
        m = self.encoder(inputs)
        return p, self.metadata_model(m)

class TransPhaseNet(tf.keras.Model):

    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 transformer_sizes=[64],
                 att_type='additive',
                 initializer='glorot_normal',
                 name='PhaseNet'):
        super(TransPhaseNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.transformer_sizes = transformer_sizes
        self.att_type = att_type

        if filters is None:
            self.filters = [4, 8, 16, 32]
        else:
            self.filters = filters

        if kernelsizes is None:
            self.kernelsizes = [7, 7, 7, 7]
        else:
            self.kernelsizes = kernelsizes

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        ### [First half of the network: downsampling inputs] ###

        def block_transformer(f, width, x):
            att, w = SeqSelfAttention(return_attention=True,
                                      attention_width=width,
                                      attention_type=self.att_type)(x)
            att = tfl.Add()([x, att])
            norm = tfl.LayerNormalization()(att)
            ff = tf.keras.Sequential([tfl.Dense(f, activation='relu', kernel_regularizer='l2'),
                                      tfl.Dropout(self.dropout_rate),
                                      tfl.Dense(norm.shape[2]),
                                      ])(norm)
            ff_add = tfl.Add()([norm, ff])
            norm_out = tfl.LayerNormalization()(ff_add)
            return norm_out, w

        # Entry block
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=2,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation("relu")(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, filters in enumerate(self.filters[1:]):
            x = tfl.Activation("relu")(x)
            x = tfl.Conv1D(filters, self.kernelsizes[i+1], padding="same",
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer,
                           )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation("relu")(x)
            x = tfl.Dropout(self.dropout_rate)(x)

            x = tfl.Conv1D(filters, self.kernelsizes[i+1], padding="same",
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer,
                           )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.MaxPooling1D(4, strides=2, padding="same")(x)

            # Project residual
            residual = tfl.Conv1D(filters, 1, strides=2, padding="same", kernel_initializer=self.initializer,
                                  )(
                previous_block_activation
            )
            x = tfl.concatenate([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        for ts in self.transformer_sizes:
            x = tfl.LSTM(ts, return_sequences=True)(x)
            x, _ = block_transformer(ts, None, x)

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###

        for filters in self.filters[::-1]:
            x = tfl.Activation("relu")(x)
            x = tfl.Conv1DTranspose(filters, self.kernelsizes[::-1][i], padding="same",
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_initializer=self.initializer,
                                    )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation("relu")(x)
            x = tfl.Dropout(self.dropout_rate)(x)

            x = tfl.Conv1DTranspose(filters, self.kernelsizes[::-1][i], padding="same",
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_initializer=self.initializer,
                                    )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.UpSampling1D(2)(x)

            # Project residual
            residual = tfl.UpSampling1D(2)(previous_block_activation)
            residual = tfl.Conv1D(filters, 1, padding="same",
                                  kernel_regularizer=self.kernel_regularizer,
                                  kernel_initializer=self.initializer,
                                  )(residual)
            x = tfl.concatenate([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        to_crop = x.shape[1] - input_shape[1]
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        x = tfl.Cropping1D((of_start, of_end))(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            outputs = tfl.Conv1D(self.num_classes,
                                 1,
                                 padding="same",
                                 activation=self.output_activation)(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

class TransPhaseNetMetadata(TransPhaseNet):
    def __init__(self, num_outputs, ph_kw):
        super(TransPhaseNetMetadata, self).__init__(**ph_kw)
        self.metadata_model = tf.keras.Sequential([tfl.Flatten(),
                                                   tfl.Dense(128, activation='relu'),
                                                   tfl.Dense(num_outputs)])
    def call(self, inputs):
        p = self.model(inputs)
        m = self.encoder(inputs)
        return p, self.metadata_model(m)

class ScatNet(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 j=[4,6,8],
                 q=[8,2,1],
                 k=7,
                 pooling_type='avg',
                 decimation=2,
                 pooling_size=1024,
                 eps=1e-3,
                 n_pca=5,
                 eps_log=0.001,
                 n_clusters=10,
                 moving_pca=0.9,
                 clustering_method='GMM',
                 name='scatnet',
                 loss_weights=(1.0, 1.0),
                 **filters_kw
                 ):
        super(ScatNet, self).__init__(name=name)

        self.batch_size = input_shape[0]
        self.n_pca = n_pca
        self.eps_log = eps_log
        self.moving_pca = moving_pca
        self.n_clusters = n_clusters
        self.eps = eps
        self.loss_weights = loss_weights
        self.clustering_method = clustering_method

        depth = len(j)
        self.ls = [Scattering(batch_size=self.batch_size,
                             index=0,
                             j=j,
                             q=q,
                             k=k,
                             decimation=decimation,
                             pooling=pooling_size,
                             pooling_type=pooling_type,
                             **filters_kw)]

        for i in range(1,depth):
            layer = Scattering(batch_size=self.batch_size,
                               index=i,
                             j=j,
                             q=q,
                             k=k,
                             decimation=decimation,
                             pooling=pooling_size,
                             pooling_type=pooling_type,
                             **filters_kw)

            self.ls.append(layer)

        if pooling_type == 'max':
            self.global_pool = tf.keras.layers.GlobalMaxPooling1D()
        else:
            self.global_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.cluster_loss_tracker = tf.keras.metrics.Mean(name="cluster_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.cluster_loss_tracker,
        ]

    def forward(self, x, training=False):
        def renorm(child, parent, epsilon=1e-3):
            # Extract all shapes.
            batch_size, *_, samples = child.shape
            if epsilon > 0:
                s = tf.experimental.numpy.swapaxes(child,1,2) / (tf.expand_dims(parent, -2) + epsilon)
                batch_size, *_, samples = s.shape
                return tf.reshape(s, [batch_size, -1, samples])
            else:
                return tf.reshape(child, [batch_size, -1, samples])

        x = tf.experimental.numpy.swapaxes(x, 1, 2)
        us, ss = [], []
        for layer in self.ls:
            u, s = layer(x)
            x = u
            us.append(u)
            ss.append(s)

        r = []
        for i in range(1, len(ss)):
            r.append(renorm(ss[i], ss[i-1], self.eps))

        sx = tf.concat(r, axis=1)

        sx = tf.math.log(sx + self.eps_log)
        sx = self.global_pool(sx)

        #sx = tf.reshape(sx, (self.batch_size, -1))
        #sx -= tf.math.reduce_mean(sx, axis=0, keepdims=True)

        return sx

    def predict_step(self, data):
        if type(data) == tuple:
            data, _ = data
        x = data
        proj = self.forward(x, training=False)
        if self.clustering_method == 'GMM':
            sample, prob, mean, logvar = self.clustering(tf.ones((proj.shape[0],1)))
        else:
            dist, assignment = self.clustering(proj)
            prob = tf.nn.softmax(-dist)

        return proj, prob

    def train_step(self, data):
        if type(data) == tuple:
            data, _ = data

        x = data

        with tf.GradientTape() as tape:

            proj = self.forward(x, training=True)
            scat_layers_loss = self.loss_weights[0] * tf.reduce_sum([sum(layer.losses) for layer in self.ls])

            if not hasattr(self, 'clustering'):
                if self.clustering_method == 'GMM':
                    self.clustering = GMM(proj.shape[1], self.n_clusters)
                else:
                    self.clustering = KMeans(proj.shape[1], self.n_clusters)

            if self.clustering_method == 'GMM':
                sample, prob, mean, logvar = self.clustering(tf.ones((proj.shape[0],1)))
                log_likelihood = self.clustering.log_likelihood(proj, prob, mean, logvar) #Should it be proj or sample?
            else:
                dist, _ = self.clustering(proj)
                log_likelihood = tf.reduce_mean(dist)

            cluster_loss = self.loss_weights[1] * log_likelihood
            total_loss = scat_layers_loss + cluster_loss

        w = []
        for layer in self.ls:
            w += layer.variables
        w += self.clustering.variables

        grads = tape.gradient(total_loss, w)
        self.optimizer.apply_gradients(zip(grads, w))
        self.total_loss_tracker.update_state(total_loss)
        self.cluster_loss_tracker.update_state(cluster_loss)
        self.reconstruction_loss_tracker.update_state(scat_layers_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "cluster_loss": self.cluster_loss_tracker.result(),
        }


class ScatNetV2(tf.keras.Model):
    def __init__(self,
                 layer_properties,
                 bins=128,
                 sampling_rate=1.0,
                 pool_type='avg',
                 data_format='channels_last',
                 combine=False,
                 name='scatnet'):
        super(ScatNetV2, self).__init__(name=name)

        self.banks = [ScatteringV2(bins=bins,
                                   sampling_rate=sampling_rate,
                                   **p) for p in layer_properties]
        self.sampling_rate = sampling_rate
        self.data_format = data_format
        self.combine = combine

        if pool_type == 'avg':
            self.pool = lambda x: tf.math.reduce_mean(x, axis=-1)
        elif pool_type == 'max':
            self.pool = lambda x: tf.math.reduce_max(x, axis=-1)
        else:
            self.pool = lambda x: x

    def __call__(self, inputs, training=False):
        if isinstance(inputs, tuple):
            inputs, _ = inputs
        x = inputs

        if self.data_format == 'channels_last':
            x = tf.transpose(x, [0, 2, 1])

        output = []

        for bank in self.banks:
            scalogram = bank(x)
            x = scalogram
            output.append(self.pool(scalogram))

        if self.combine:
            output = [tf.reshape(out, (x.shape[0], -1, x.shape[-1])) for out in output]
            output = tf.concat(output, axis=1)
            if self.data_format == 'channels_last':
                output = tf.transpose(output, [0, 2, 1])

        return output

    @property
    def depth(self):
        return len(self.banks)
