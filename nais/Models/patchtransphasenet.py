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

from nais.Models import PhaseNet
from nais.Layers import DynamicConv1D, ResnetBlock1D, TransformerBlock, Resampling1D

class Patches(tfl.Layer):
    def __init__(self, patch_size, patch_stride):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_stride, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        
        return patches

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
                 att_type='downstep',
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
            att_type (str, optional): if the attention should work during downstep or across. 
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
        
        if residual_attention is None:
            self.residual_attention = [16, 16, 16, 16]
        else:
            self.residual_attention = residual_attention
        
        if patch_sizes is None: 
            self.patch_sizes = [8, 8, 8, 8]
        else:
            self.patch_sizes = patch_sizes
            
        if patch_strides is None:
            self.patch_strides = [8, 8, 8, 8]
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
        out = x
        
        x = tfl.Bidirectional(tfl.LSTM(ra, return_sequences=True))(x)
        x = tfl.Conv1D(ra, 1, padding='same')(x)
        
        out_length = x.shape[1]
        
        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        
        x = Patches(patch_size=npatch, patch_stride=spatch)(x)
        y = Patches(patch_size=npatch, patch_stride=spatch)(y)
        
        if self.intra_patch_attention:
            
            x_shape = x.shape[1:]
            y_shape = y.shape[1:]
            
            x = tfl.Reshape((x.shape[-1], -1))(x)
            y = tfl.Reshape((y.shape[-1], -1))(y)
            
            x = TransformerBlock(num_heads=8,
                                embed_dim=x.shape[-1],
                                ff_dim=ra*4,
                                rate=self.dropout_rate)(x)
            y = TransformerBlock(num_heads=8,
                                embed_dim=y.shape[-1],
                                ff_dim=ra*4,
                                rate=self.dropout_rate)(y)
                    
            x = tfl.Reshape(x_shape)(x)
            y = tfl.Reshape(y_shape)(y)

                
        x = tfl.Reshape((x.shape[1], -1))(x)
        y = tfl.Reshape((y.shape[1], -1))(y)
        
        pos = tfl.Embedding(input_dim=x.shape[1], output_dim=x.shape[2])(tf.range(start=0, limit=x.shape[1], delta=1))
        x += pos
        
        att = TransformerBlock(num_heads=8,
                               embed_dim=x.shape[-1],
                               ff_dim=ra*4,
                               rate=self.dropout_rate)([x,y])
        
        att = tfl.Conv1D(ra, 1, padding='same')(att)
        att = Resampling1D(out_length)(att)
        
        x = crop_and_concat(out, att)
        
        return x

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
                x = self._att_block(x, 
                                    skips[-1], 
                                    self.residual_attention[i], 
                                    self.patch_sizes[i], 
                                    self.patch_strides[i])
            skips.append(x)

        if self.residual_attention[-1] > 0:
            x = self._att_block(x, x, self.residual_attention[-1], self.patch_sizes[-1], self.patch_strides[-1])

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        
        for i in range(1, len(self.filters)):
            x = self._up_block(self.filters[::-1][i], self.kernelsizes[::-1][i], x)
            
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                x = self._att_block(x, 
                                    skips[::-1][i],
                                    self.residual_attention[::-1][i], 
                                    self.patch_sizes[::-1][i], 
                                    self.patch_strides[::-1][i])

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
    
