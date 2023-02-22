
import tensorflow.keras.layers as tfl
import tensorflow as tf 

class DynamicConv1D(tfl.Layer):
    def __init__(self, 
                 filters, 
                 kernelsize, 
                 num_layers=3, 
                 activation='relu',
                 dropout=0.0,
                 **kwargs):
        """1D dynamic convolution 
            Based on https://arxiv.org/abs/1912.03458
        Args:
            filters (int): number of filters
            kernelsize (int): kernel size of filters
            num_layers (int, optional): number of dynamic layers. Defaults to 3.
            activation (str, optional): activation function. Defaults to 'relu'.
            dropout (float, optional): dropout. Defaults to 0.0.
        """

        super(DynamicConv1D, self).__init__(**kwargs)
        
        self.attention = tf.keras.Sequential([tfl.GlobalAveragePooling1D(),
                                              tfl.Dense(filters, activation='relu'),
                                              tfl.Dropout(dropout),
                                              tfl.Dense(num_layers, activation='softmax')])
        self.convs = [tfl.Conv1D(filters, kernelsize, padding='same', **kwargs) for _ in range(num_layers)]
        self.out_layer = tf.keras.Sequential([tfl.Conv1D(filters, kernelsize, padding='same'),
                                              tfl.BatchNormalization(),
                                              tfl.Activation(activation),
                                              tfl.Dropout(dropout)])
        self.dot = tfl.Dot(axes=-1)
        
    def call(self, inputs):
        att = self.attention(inputs)
        conv = tf.stack([l(inputs) for l in self.convs], axis=-1)
        out = self.dot([att, conv])
        
        return self.out_layer(out)
