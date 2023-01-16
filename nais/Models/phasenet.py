""" 
Various variations of PhaseNet 
Author: Erik 
Email: erik@norsar.no
"""

import tensorflow as tf 
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K
import numpy as np
from nais.Layers import ResnetBlock1D, SeqSelfAttention

class PhaseNet(tf.keras.Model):
    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 initializer='glorot_normal',
                 name='PhaseNet'):
        """Adapted to 1D from https://keras.io/examples/vision/oxford_pets_image_segmentation/

        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
        """
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

class PhaseNetMetadata(PhaseNet):
    def __init__(self, num_outputs=None, metadata_model=None, ph_kw=None):
        """Provides a wrapper for PhaseNet with a metadata output, eg., when learning back azimuth.

        Args:
            num_outputs (int): Number of numerical metadata to learn. 
            ph_kw (dict): Args. for PhaseNet. 
        """
        super(PhaseNetMetadata, self).__init__(**ph_kw)
        if metadata_model is None:
            self.metadata_model = tf.keras.Sequential([tfl.Flatten(),
                                                   tfl.Dense(128, activation='relu'),
                                                   tfl.Dense(num_outputs)])
        else:
            self.metadata_model = metadata_model
            
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
                 residual_attention=False,
                 name='TransPhaseNet'):
        """Adds self-attention in bottleneck of PhaseNet. 

        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): List of kernel sizes. Defaults to None.
            output_activation (str, optional): Activation for output, eg., 'softmax' for multiclass. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regulizers.Regulizer, optional): regularizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            transformer_sizes (list, optional): List of width of self-attention. Defaults to [64].
            att_type (str, optional): Self-attention type. Defaults to 'additive'. 'multiplicative' as alternative. 
            initializer (tf.keras.initializers.Initializer, optional): layer initializers. Defaults to 'glorot_normal'.
            name (str, optional): Name of Model. Defaults to 'TransPhaseNet'.
        """
        super(TransPhaseNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.transformer_sizes = transformer_sizes
        self.residual_attention = residual_attention
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
            x = tfl.Bidirectional(tfl.LSTM(ts, return_sequences=True))(x)
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
            if self.residual_attention:
                residual, _ = block_transformer(residual.shape[-1], None, residual)
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
    def __init__(self, num_outputs=None, metadata_model=None, ph_kw=None):
        """Provides a wrapper for TransPhaseNet with a metadata output, eg., when learning back azimuth.

        Args:
            num_outputs (int): Number of numerical metadata to learn. 
            ph_kw (dict): Args. for TransPhaseNet. 
        """
        super(TransPhaseNetMetadata, self).__init__(**ph_kw)
        if metadata_model is None:
            self.metadata_model = tf.keras.Sequential([tfl.Flatten(),
                                                   tfl.Dense(128, activation='relu'),
                                                   tfl.Dense(num_outputs)])
        else:
            self.metadata_model = metadata_model
    def call(self, inputs):
        p = self.model(inputs)
        m = self.encoder(inputs)
        return p, self.metadata_model(m)
    