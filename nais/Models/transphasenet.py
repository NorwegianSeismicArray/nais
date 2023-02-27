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

from nais.Layers import DynamicConv1D, TransformerBlock, ResnetBlock1D

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
                 activation='relu',
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
        self.activation = activation

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
        
        # Entry block
        
        x = ResnetBlock1D(self.filters[0], 
                          self.kernelsizes[0], 
                          activation=self.activation, 
                          dropout=self.dropout_rate,
                          match_filters=not self.filters[0] == inputs.shape[-1])(inputs)

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i in range(1, len(self.filters)):
            x = ResnetBlock1D(self.filters[i], 
                              self.kernelsizes[i], 
                              activation=self.activation, 
                              dropout=self.dropout_rate,
                              match_filters=not self.filters[i] == x.shape[-1])(x)
            
            x = self.pool_layer(4, strides=2, padding="same")(x)
            skips.append(x)

        if self.residual_attention[-1] > 0:
            x = tfl.Bidirectional(tfl.LSTM(self.residual_attention[-1], return_sequences=True), merge_mode='ave')(x)
            att = TransformerBlock(num_heads=8,
                                  embed_dim=self.residual_attention[-1],
                                  ff_dim=self.residual_attention[-1],
                                  rate=self.dropout_rate)(x)
            x = crop_and_concat(x, att)

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        
        for i in range(1, len(self.filters)):
            x = ResnetBlock1D(self.filters[::-1][i], 
                              self.kernelsizes[::-1][i], 
                              activation=self.activation, 
                              dropout=self.dropout_rate,
                              match_filters=not self.filters[::-1][i] == x.shape[-1])(x)
            x = tfl.UpSampling1D(2)(x)
            
            if self.residual_attention[::-1][i] > 0:
                x = tfl.Bidirectional(tfl.LSTM(self.residual_attention[::-1][i], return_sequences=True), merge_mode='ave')(x)
                att = TransformerBlock(num_heads=8,
                                  embed_dim=self.residual_attention[::-1][i],
                                  ff_dim=self.residual_attention[::-1][i],
                                  rate=self.dropout_rate)([x, skips[::-1][i]])
                x = crop_and_concat(x, att)

        to_crop = x.shape[1] - input_shape[1]
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        x = tfl.Cropping1D((of_start, of_end))(x)
        
        #Exit block
        x = ResnetBlock1D(self.filters[0], 
                          self.kernelsizes[0], 
                          activation=self.activation, 
                          dropout=self.dropout_rate,
                          match_filters=self.filters[0] == x.shape[-1])(x)

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
    
