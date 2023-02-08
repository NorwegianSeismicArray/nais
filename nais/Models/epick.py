""" 
Various variations of PhaseNet 
Author: Erik 
Email: erik@norsar.no
"""

import tensorflow as tf 
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K
import numpy as np
from nais.utils import crop_and_concat
    
class EPick(tf.keras.Model):
    def __init__(self,
                 num_classes=2,
                 output_layer=None,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 att_type='additive',
                 initializer='glorot_normal',
                 residual_attention=None,
                 name='EPick'):
        """
        https://arxiv.org/abs/2109.02567
        
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
        super(EPick, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.residual_attention = residual_attention
        self.att_type = att_type
        self.output_layer = output_layer

        if filters is None:
            self.filters = [16, 16, 16, 16]
        else:
            self.filters = filters
            
        if residual_attention is None:
            self.residual_attention = [16, 16, 16, 16, 16]
        else:
            self.residual_attention = residual_attention

        if kernelsizes is None:
            self.kernelsizes = [7, 7, 7, 7]
        else:
            self.kernelsizes = kernelsizes

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        ### [First half of the network: downsampling inputs] ###

        def down_block(f, ks, x):
            x = tfl.Conv1D(f, ks, padding="same",
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer,
                           )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation("relu")(x)
            x = tfl.Dropout(self.dropout_rate)(x)
            x = tfl.MaxPooling1D(4, strides=2, padding='same')(x)
            return x

        def up_block(f, ks, x, upsample=True):
            x = tfl.Conv1DTranspose(f, ks, padding="same",
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_initializer=self.initializer,
                                    )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation("relu")(x)
            x = tfl.Dropout(self.dropout_rate)(x)
            if upsample:
                x = tfl.UpSampling1D(2)(x)
            return x
            
        # Entry block
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation("relu")(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        
        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for ks, f in zip(self.kernelsizes[1:], self.filters[1:]):
            x = down_block(f, ks, x) 
            skips.append(x)
        
        attentions = []
        for i, skip in enumerate(skips):
            if self.residual_attention[i] <= 0:
                att = skip
            elif i == 0:
                att = tfl.MultiHeadAttention(num_heads=8, 
                                             key_dim=self.residual_attention[i],)(skip, skip, return_attention_scores=False)
            else:
                tmp = []
                z = skips[i]
                for j, skip2 in enumerate(skips[:i]):
                    if self.residual_attention[j] <= 0:
                        att = tfl.Conv1D(self.filters[j], 3, activation='relu', padding='same')(z)
                    else:
                        att = tfl.MultiHeadAttention(num_heads=8, 
                                                     key_dim=self.residual_attention[j])(z, skip2, return_attention_scores=False)
                    tmp.append(att)
                att = tfl.Concatenate()(tmp)
            attentions.append(att)
            
        x = crop_and_concat(x, attentions[-1])
        self.encoder = tf.keras.Model(inputs, x)
            
        i = len(self.filters) - 1
        for f, ks in zip(self.filters[::-1][:-1], self.kernelsizes[::-1][:-1]):
            x = up_block(f, ks, x, upsample = i != 0)
            x = crop_and_concat(x, attentions[i-1])
            i -= 1
        
        to_crop = x.shape[1] - input_shape[1]
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        x = tfl.Cropping1D((of_start, of_end))(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
            outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
        elif self.output_layer is not None:
            outputs = self.output_layer(x)
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

class EPickMetadata(EPick):
    def __init__(self, num_outputs=None, metadata_model=None, ph_kw=None):
        """Provides a wrapper for TransPhaseNet with a metadata output, eg., when learning back azimuth.

        Args:
            num_outputs (int): Number of numerical metadata to learn. 
            ph_kw (dict): Args. for TransPhaseNet. 
        """
        super(EPickMetadata, self).__init__(**ph_kw)
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
    