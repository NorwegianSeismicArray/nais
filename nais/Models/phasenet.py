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


class PhaseNet(tf.keras.Model):
    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 pool_type='max',
                 activation='relu',
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
        self.activation = activation

        if filters is None:
            self.filters = [4, 8, 16, 32]
        else:
            self.filters = filters

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

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, filters in enumerate(self.filters):
            x = tfl.Conv1D(filters, self.kernelsizes[i], padding="same",
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializer,
                           )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation(self.activation)(x)
            x = tfl.Dropout(self.dropout_rate)(x)

            x = self.pool_layer(4, strides=2, padding="same")(x)

            skips.append(x)
            
        skips = skips[:-1]

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        skips = skips[::-1]
        
        for i, filters in enumerate(self.filters[::-1]):
            x = tfl.Conv1DTranspose(filters, self.kernelsizes[::-1][i], padding="same",
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_initializer=self.initializer,
                                    )(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation(self.activation)(x)
            x = tfl.Dropout(self.dropout_rate)(x)

            x = tfl.UpSampling1D(2)(x)

            x = crop_and_concat(x, skips[i])

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
    
    
from nais.Layers import ResidualConv1D, ResidualConv1DTranspose, ResnetBlock1D

class ResidualPhaseNet(tf.keras.Model):
    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 pool_type='max',
                 activation='relu',
                 initializer='glorot_normal',
                 name='ResidualPhaseNet'):
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
        super(ResidualPhaseNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.activation = activation

        if filters is None:
            self.filters = [4, 8, 16, 32]
        else:
            self.filters = filters

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

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        previous_block_activation = x  # Set aside residual

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, (f, ks) in enumerate(zip(self.filters, self.kernelsizes)):
            x = ResnetBlock1D(f, ks, activation=self.activation, dropout=self.dropout_rate)(x)
            x = self.pool_layer(4, strides=2, padding="same")(x)
            skips.append(x)
            
        skips = skips[:-1]

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        skips = skips[::-1]
        
        for i, (f, ks) in enumerate(zip(self.filters[::-1], self.kernelsizes[::-1])):
            x = ResnetBlock1D(f, ks, activation=self.activation, dropout=self.dropout_rate)(x)
            x = tfl.UpSampling1D(2)(x)

            x = crop_and_concat(x, skips[i])

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
    
    
    
class ResidualPhaseNetMetadata(ResidualPhaseNet):
    def __init__(self, num_outputs=None, metadata_model=None, ph_kw=None):
        """Provides a wrapper for PhaseNet with a metadata output, eg., when learning back azimuth.

        Args:
            num_outputs (int): Number of numerical metadata to learn. 
            ph_kw (dict): Args. for PhaseNet. 
        """
        super(ResidualPhaseNetMetadata, self).__init__(**ph_kw)
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