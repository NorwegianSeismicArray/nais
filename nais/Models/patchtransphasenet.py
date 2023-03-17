""" 
Various variations of TransPhaseNet 
Author: Erik 
Email: erik@norsar.no
"""

import tensorflow as tf 
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K
import numpy as np
from nais.utils import crop_and_concat, crop_and_add

from nais.Models import PhaseNet
from nais.Layers import DynamicConv1D, ResnetBlock1D, TransformerBlock, Resampling1D, PatchTransformerBlock, ResidualConv1D

class PatchTransPhaseNet(PhaseNet):
    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 initializer='glorot_normal',
                 residual_attention=None,
                 patch_sizes=None,
                 patch_strides=None,
                 intra_patch_attention=False,
                 pool_type='max',
                 att_type='across',
                 rnn_type='lstm',
                 additive_att=True,
                 stacked_layer=4,
                 activation='relu',
                 name='PatchTransPhaseNet'):
        """Adapted to 1D from https://keras.io/examples/vision/oxford_pets_image_segmentation/

        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            residual_attention (list: optional): list of residual attention sizes, one longer that filters. 
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
            att_type (str, optional): if the attention should work during downstep or across (self attention). 
        """
        super(PatchTransPhaseNet, self).__init__(num_classes=num_classes, 
                                              filters=filters, 
                                              kernelsizes=kernelsizes, 
                                              output_activation=output_activation, 
                                              kernel_regularizer=kernel_regularizer, 
                                              dropout_rate=dropout_rate,
                                              pool_type=pool_type, 
                                              activation=activation, 
                                              initializer=initializer, 
                                              name=name)
        self.att_type = att_type
        self.intra_patch_attention = intra_patch_attention
        self.rnn_type = rnn_type
        self.stacked_layer = stacked_layer
        self.additive_att = additive_att
        
        if residual_attention is None:
            self.residual_attention = [16, 16, 16, 16]
        else:
            self.residual_attention = residual_attention
        
        if patch_sizes is None: 
            self.patch_sizes = [8, 8, 8, 8]
        else:
            self.patch_sizes = patch_sizes
            
        if patch_strides is None:
            self.patch_strides = self.patch_sizes
        else:
            self.patch_strides = patch_strides
    
    def _down_block(self, f, ks, x):
        x = ResnetBlock1D(f, 
                        ks, 
                        activation=self.activation, 
                        dropout=self.dropout_rate)(x)    
        x = self.pool_layer(4, strides=2, padding="same")(x)
        return x
    
    def _up_block(self, f, ks, x):
        x = ResnetBlock1D(f, 
                        ks, 
                        activation=self.activation, 
                        dropout=self.dropout_rate)(x)
        x = tfl.UpSampling1D(2)(x)
        return x

    def _att_block(self, x, y, ra, npatch, spatch):
 
        if self.rnn_type == 'lstm':
            x = tfl.Bidirectional(tfl.LSTM(ra, return_sequences=True))(x)
        elif self.rnn_type == 'causal':
            x1 = ResidualConv1D(ra, 3, stacked_layer=self.stacked_layer, causal=True)(x)
            x2 = ResidualConv1D(ra, 3, stacked_layer=self.stacked_layer, causal=True)(tf.reverse(x, axis=[1]))
            x = tf.concat([x1, tf.reverse(x2, axis=[1])], axis=-1)
        else:
            raise NotImplementedError('rnn type:' + self.rnn_type + ' is not supported')
        
        x = tfl.Conv1D(ra, 1, padding='same')(x) 
        
        if npatch > 1:
            att = PatchTransformerBlock(npatch,
                                        num_heads=8,
                                        embed_dim=ra*npatch,
                                        ff_dim=ra*npatch*4,
                                        rate=self.dropout_rate)([x,y])
        else:
            att = TransformerBlock(num_heads=8,
                                embed_dim=ra,
                                ff_dim=ra*4,
                                rate=self.dropout_rate)([x,y])
                        
        return att

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])
        
        # Entry block
        
        x = ResnetBlock1D(self.filters[0], 
                          self.kernelsizes[0], 
                          activation=self.activation, 
                          dropout=self.dropout_rate)(inputs)

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i in range(1, len(self.filters)):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            if self.residual_attention[i] > 0 and self.att_type == 'downstep':
                att = self._att_block(x, 
                                    skips[-1], 
                                    self.residual_attention[i], 
                                    self.patch_sizes[i], 
                                    self.patch_strides[i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[i], 1, padding='same')(x)
                    
            skips.append(x)

        if self.residual_attention[-1] > 0:
            att = self._att_block(x, x, self.residual_attention[-1], self.patch_sizes[-1], self.patch_strides[-1])
            if self.additive_att:
                x = crop_and_add(x, att)
            else:
                x = crop_and_concat(x, att)
                x = tfl.Conv1D(self.filters[-1], 1, padding='same')(x)

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        
        for i in range(1, len(self.filters)):
            x = self._up_block(self.filters[::-1][i], self.kernelsizes[::-1][i], x)
            
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(skips[::-1][i],
                                    skips[::-1][i],
                                    self.residual_attention[::-1][i], 
                                    self.patch_sizes[::-1][i], 
                                    self.patch_strides[::-1][i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[::-1][i], 1, padding='same')(x)

        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
        
        #Exit block
        x = tfl.Conv1D(self.filters[0], 
                       self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='exit')(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
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
    
class PatchTransPhaseNetMetadata(PatchTransPhaseNet):
    def __init__(self, num_outputs=None, metadata_model=None, ph_kw=None):
        """Provides a wrapper for TransPhaseNet with a metadata output, eg., when learning back azimuth.

        Args:
            num_outputs (int): Number of numerical metadata to learn. 
            ph_kw (dict): Args. for TransPhaseNet. 
        """
        super(PatchTransPhaseNetMetadata, self).__init__(**ph_kw)
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
    
