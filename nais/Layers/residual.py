import tensorflow.keras.layers as tfl 
import tensorflow as tf 

class ResidualConv1D(tfl.Layer):
    """
    1D residual convolution 

    Args:
        filters (int): number of filters
        kernel_size (int): size of filters 
        stacked_layers (int): number of stacked layers 
    """
    def __init__(self, filters=32, kernel_size=3, stacked_layer=1):
        super(ResidualConv1D, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stacked_layer = stacked_layer

    def build(self, input_shape):
        self.sigmoid_layers = []
        self.tanh_layers = []
        self.conv_layers = []

        for dilation_rate in [2 ** i for i in range(self.stacked_layer)]:
            self.sigmoid_layers.append(
                tfl.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same',
                                    activation='sigmoid'))
            self.tanh_layers.append(
                tfl.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same',
                                    activation='tanh'))
            self.conv_layers.append(tfl.Conv1D(self.filters, 1, padding='same'))

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


class ResidualConv1DTranspose(tfl.Layer):
    """Inverse 1D residual convolution

    Args:
        filters (int): number of filters 
        kernel_size (int): size of filters 
        stacked_layer (int): number of stacked layers 
    """
    
    def __init__(self, filters=32, kernel_size=3, stacked_layer=1):
        super(ResidualConv1DTranspose, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stacked_layer = stacked_layer

    def build(self, input_shape):
        self.sigmoid_layers = []
        self.tanh_layers = []
        self.conv_layers = []

        for dilation_rate in [2 ** i for i in range(self.stacked_layer)]:
            self.sigmoid_layers.append(tfl.Conv1DTranspose(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same', activation='sigmoid'))
            self.tanh_layers.append(tfl.Conv1DTranspose(self.filters, self.kernel_size, dilation_rate=dilation_rate, padding='same', activation='tanh'))
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
    """1D resnet block

    Args:
        filters (int): number of filters 
        kernel_size (int): size of filters 
        activation (str): layer activation
        dropout (float): dropout fraction 
    """
    def __init__(self, filters, kernelsize, activation='relu', dropout=0.1, **kwargs):
        super(ResnetBlock1D, self).__init__()
        self.conv1 = tfl.Conv1D(filters, kernelsize, activation=None, padding='same', **kwargs)
        self.conv2 = tfl.Conv1D(filters, kernelsize, activation=None, padding='same', **kwargs)
        self.dropout1 = tfl.SpatialDropout1D(dropout)
        self.bn1 = tfl.BatchNormalization()
        self.bn2 = tfl.BatchNormalization()
        self.add = tfl.Add()
        self.relu = tfl.Activation(activation)

    @tf.function
    def call(self, inputs, training=None):
        fx = self.conv1(inputs)
        fx = self.bn1(fx)
        fx = self.relu(fx)
        fx = self.dropout1(fx)
        fx = self.conv2(fx)
        x = self.add([inputs, fx])
        x = self.bn2(x)
        x = self.relu(x)
        return x
