
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
        super(DynamicConv1D, self).__init__(**kwargs)
        
        self.attention = tf.keras.Sequential([tfl.GlobalAveragePooling1D(), 
                                              tfl.Dense(filters, activation='relu'),
                                              tfl.Dropout(dropout),
                                              tfl.Dense(num_layers, activation='softmax')])
        self.convs = [tfl.Conv1D(filters, kernelsize, padding='same') for _ in range(num_layers)]
        self.out_layer = tf.keras.Sequential([tfl.Conv1D(filters, kernelsize, padding='same'),
                                              tfl.BatchNormalization(),
                                              tfl.Activation(activation),
                                              tfl.Dropout(dropout)])
        self.dot = tfl.Dot(axes=-1)
        
    def call(self, inputs):
        att = self.attention(inputs)
        conv = tf.stack([l(inputs) for l in self.convs], axis=-1)
        out = self.dot([att, conv])
        
        return out 
