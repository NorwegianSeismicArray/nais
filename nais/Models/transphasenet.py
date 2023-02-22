""" 
Various variations of TransPhaseNet 
Author: Erik 
Email: erik@norsar.no
"""

import tensorflow as tf 
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K
import numpy as np
from nais.utils import crop_and_concat

from nais.Layers import DynamicConv1D, TransformerBlock

class TransPhaseNet(tf.keras.Model):
    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 att_type='additive',
                 initializer='glorot_normal',
                 residual_attention=None,
                 pool_type='max',
                 name='TransPhaseNet'):
        """Adapted to 1D from https://keras.io/examples/vision/oxford_pets_image_segmentation/

        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            residual_attention (list: optional): list of residual attention sizes, one longer that filters. 
            att_type (str): dot or concat
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
        """
        super(TransPhaseNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.residual_attention = residual_attention
        self.att_type = att_type

        if filters is None:
            self.filters = [4, 8, 16, 32]
        else:
            self.filters = filters
            
        if residual_attention is None:
            self.residual_attention = [0, 0, 0, 0, 0]
        else:
            self.residual_attention = residual_attention

        if kernelsizes is None:
            self.kernelsizes = [7, 7, 7, 7]
        else:
            self.kernelsizes = kernelsizes
            
        if pool_type == 'max':
            self.pool_layer = tfl.MaxPooling1D
        else:
            self.pool_layer = tfl.AveragePooling1D

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])
        
        def block_transformer(f, width, query, value):
            lstm_block = tf.keras.Sequential([tfl.Bidirectional(tfl.LSTM(f, return_sequences=True)),
                                              tfl.Conv1D(f, 1, 
                                                         padding='same', 
                                                         kernel_regularizer=self.kernel_regularizer)])
            Q, V = lstm_block(query), lstm_block(value)
            return TransformerBlock(num_heads=8, embed_dim=f, ff_dim=f, rate=seld.dropout_rate)([Q, V])

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation("relu")(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        previous_block_activation = x  # Set aside residual

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, filters in enumerate(self.filters):
            x = tfl.Activation("relu")(x)
            x = tfl.Conv1D(filters, self.kernelsizes[i], padding="same",
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer,
                           )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation("relu")(x)
            x = tfl.Dropout(self.dropout_rate)(x)

            x = tfl.Conv1D(filters, self.kernelsizes[i], padding="same",
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer,
                           )(x)
            x = tfl.BatchNormalization()(x)

            x = self.pool_layer(4, strides=2, padding="same")(x)

            # Project residual
            residual = tfl.Conv1D(filters, 1, strides=2, padding="same", kernel_initializer=self.initializer,
                                  )(
                previous_block_activation
            )
            x = tfl.concatenate([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
            skips.append(x)

        if self.residual_attention[-1] > 0:
            x, _ = block_transformer(self.residual_attention[-1], None, x, x)

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        skips = skips[:-1]
        
        c, f = range(len(self.residual_attention)-2, -1, -1), self.filters[::-1]
        
        for i, filters in zip(c, f):
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
            
            if self.residual_attention[i] > 0:
                att, _ = block_transformer(self.residual_attention[i], None, x, skips[i])
                x = crop_and_concat(x, att)
            x = tfl.concatenate([x, residual]) # Add back residual
            previous_block_activation = x  # Set aside next residual

        to_crop = x.shape[1] - input_shape[1]
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        x = tfl.Cropping1D((of_start, of_end))(x)
        
        #Exit block
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='exit')(x)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation("relu")(x)
        x = tfl.Dropout(self.dropout_rate)(x) 

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
            outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
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
    