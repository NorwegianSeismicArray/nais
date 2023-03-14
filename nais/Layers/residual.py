import tensorflow.keras.layers as tfl 
import tensorflow as tf 
import tensorflow.keras.backend as K

def mish(x):
	return tfl.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
get_custom_objects().update({'mish': Activation(mish)})

class ResidualConv1D(tfl.Layer):
   
    def __init__(self, 
                 filters=32, 
                 kernel_size=3, 
                 stacked_layer=1, 
                 activation='relu',
                 causal=False):
        """1D residual convolution 
        
        Args:
            filters (int): number of filters.
            kernel_size (int): size of filters .
            stacked_layers (int): number of stacked layers .
        """
        
        super(ResidualConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stacked_layer = stacked_layer
        self.causal = causal
        self.activation = activation

    def build(self, input_shape):
        self.sigmoid_layers = []
        self.tanh_layers = []
        self.conv_layers = []
        
        self.shape_matching_layer = tfl.Conv1D(self.filters, 1, padding = 'same')
        self.add = tfl.Add()
        self.final_activation = tf.keras.activations.get(self.activation)
        
        for dilation_rate in [2 ** i for i in range(self.stacked_layer)]:
            self.sigmoid_layers.append(
                tfl.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate, 
                           padding='causal' if self.causal else 'same',
                                    activation='sigmoid'))
            self.tanh_layers.append(
                tfl.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate, 
                           padding='causal' if self.causal else 'same',
                                    activation='tanh'))
            self.conv_layers.append(tfl.Conv1D(self.filters, 1, padding='same'))

    def get_config(self):
        return dict(name=self.name,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    stacked_layer=self.stacked_layer)

    def call(self, inputs):
        out = self.shape_matching_layer(inputs)
        residual_output = out
        x = inputs
        for sl, tl, cl in zip(self.sigmoid_layers, self.tanh_layers, self.conv_layers):
            sigmoid_x = sl(x)
            tanh_x = tl(x)

            x = tfl.multiply([sigmoid_x, tanh_x])
            x = cl(x)
            residual_output = tfl.add([residual_output, x])

        return self.final_activation(self.add([out, x]))

class ResidualConv1DTranspose(tfl.Layer):
    
    def __init__(self, filters=32, kernel_size=3, stacked_layer=1, causal=False):
        """Inverse 1D residual convolution

        Args:
            filters (int): number of filters .
            kernel_size (int): size of filters .
            stacked_layer (int): number of stacked layers .
        """
        super(ResidualConv1DTranspose, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stacked_layer = stacked_layer
        self.causal = causal

    def build(self, input_shape):
        self.sigmoid_layers = []
        self.tanh_layers = []
        self.conv_layers = []

        for dilation_rate in [2 ** i for i in range(self.stacked_layer)]:
            self.sigmoid_layers.append(tfl.Conv1DTranspose(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='causal' if self.causal else 'same', activation='sigmoid'))
            self.tanh_layers.append(tfl.Conv1DTranspose(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='causal' if self.causal else 'same', activation='mish'))
            self.conv_layers.append(tfl.Conv1DTranspose(self.filters, 1, padding='same'))

    def get_config(self):
        return dict(name=self.name,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    stacked_layer=self.stacked_layer)

    def call(self, inputs):
        residual_output = inputs
        x = inputs
        for sl, tl, cl in zip(self.sigmoid_layers, self.tanh_layers, self.conv_layers):
            sigmoid_x = sl(x)
            tanh_x = tl(x)

            x = tfl.multiply([sigmoid_x, tanh_x])
            x = cl(x)
            residual_output = tfl.add([residual_output, x])

        return residual_output
    
    
class ResnetBlock1D(tfl.Layer):
    def __init__(self, 
                 filters, 
                 kernelsize, 
                 activation='linear',
                 dropout=0.1, **kwargs):
        """1D resnet block

        Args:
            filters (int): number of filters .
            kernel_size (int): size of filters .
            activation (str): layer activation.
            dropout (float): dropout fraction .
        """
        super(ResnetBlock1D, self).__init__()
        self.filters = filters
        self.projection = tfl.Conv1D(filters, 1, padding='same', **kwargs)
        self.conv1 = tfl.Conv1D(filters, kernelsize, activation=None, padding='same', **kwargs)
        self.conv2 = tfl.Conv1D(filters, kernelsize, activation=None, padding='same', **kwargs)
        self.dropout1 = tfl.Dropout(dropout)
        self.bn1 = tfl.BatchNormalization()
        self.bn2 = tfl.BatchNormalization()
        self.bn3 = tfl.BatchNormalization()
        self.add = tfl.Add()
        self.relu = tfl.Activation(activation)

    def call(self, inputs, training=None):
        x = self.projection(inputs)
        fx = self.bn1(inputs)
        fx = self.conv1(fx)
        fx = self.bn2(fx)
        fx = self.relu(fx)
        fx = self.dropout1(fx)
        fx = self.conv2(fx)
        x = self.add([x, fx])
        x = self.bn3(x)
        x = self.relu(x)
        return x


    
class ResStageBlock1D(tfl.Layer):
    def __init__(self, 
                 filters, 
                 kernelsize, 
                 activation='relu', 
                 match_filters = False,
                 dropout=0.1, **kwargs):
        """1D resnet block

        Args:
            filters (int): number of filters .
            kernel_size (int): size of filters .
            activation (str): layer activation.
            dropout (float): dropout fraction .
        """
        super(ResStageBlock1D, self).__init__()
        self.filters = filters
        if match_filters:
            self.projection = tfl.Conv1D(filters, 1, padding='same', **kwargs)
        self.conv1 = tfl.Conv1D(filters, 1, activation=None, padding='same', **kwargs)
        self.conv2 = tfl.Conv1D(filters, 1, activation=None, padding='same', **kwargs)
        self.conv_bottleneck = tfl.Conv1D(filters//4, 3, activation=None, padding='same', **kwargs)
        self.dropout1 = tfl.Dropout(dropout)
        self.bn1 = tfl.BatchNormalization()
        self.bn2 = tfl.BatchNormalization()
        self.bn3 = tfl.BatchNormalization()
        self.bn4 = tfl.BatchNormalization()
        self.add = tfl.Add()
        self.relu = tfl.Activation(activation)

    def call(self, inputs, training=None):
        x = inputs 
        if x.shape[-1] != self.filters:
            x = self.projection(x)
        
        fx = self.bn1(inputs)
        fx = self.relu(fx)
        fx = self.conv1(fx)
        fx = self.bn2(fx)
        fx = self.relu(fx)
        fx = self.conv_bottleneck(fx)
        fx = self.bn3(fx)
        fx = self.relu(fx)
        fx = self.dropout1(fx)
        fx = self.conv2(fx)
        x = self.add([x, fx])
        x = self.bn4(x)
        x = self.relu(x)
        return x
