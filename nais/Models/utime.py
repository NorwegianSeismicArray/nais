
import tensorflow as tf 
import tensorflow.keras.layers as tfl 
from nais.Models import PhaseNet

class UTime(tf.keras.Model):
    def __init__(self,
                 phasenet_filters=None,
                 output_filters=None,
                 output_activation='softmax',
                 num_classes=2,
                 pool_sizes=[4],
                 pool_strides=[2],
                 dropout_rate=0.2,
                 pool_type='avg',
                 name='UTime'):
        """https://arxiv.org/abs/1910.11162

        Args:
            phasenet_filters (list, optional): list of number of filters used in PhaseNet. Defaults to None.
            output_filters (list, optional): list of number of filters after PhaseNet. Defaults to None.
            output_activation (str, optional): Activation function of last layer. Defaults to 'softmax'.
            num_classes (int, optional): number of output classes. Defaults to 2.
            pool_sizes (list, optional): list of pooling sizes. Defaults to [4].
            pool_strides (list, optional): list of pooling strides. Defaults to [2].
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            pool_type (str, optional): type of pooling, 'max' or 'avg'. Defaults to 'avg'.
            name (str, optional): model name. Defaults to 'UTime'.
            
        Raises:
            NotImplementedError: pool_type not in 'max', 'avg. 
        """
        super(UTime, self).__init__(name=name)
        self.phasenet = PhaseNet(filters=phasenet_filters, num_classes=None)
        if output_filters is None:
            self.filters = [4]
        else:
            self.filters = output_filters

        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.pool_type = pool_type
        self.output_activation = output_activation
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.phasenet.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        x = self.phasenet(inputs)
        for i, psize, pstride in enumerate(zip(self.pool_sizes, self.pool_strides)):
            if self.pool_type == 'avg':
                x = tfl.AveragePooling1D(psize, strides=pstride, padding='same')(x)
            elif self.pool_type == 'max':
                x = tfl.MaxPooling1D(psize, strides=pstride, padding='same')(x)
            else:
                raise NotImplementedError(f'pool_type={self.pool_type} is not supported.')
            if i < len(self.pool_size) - 1: #skip when on last pool layer.
                x = tfl.Conv1D(self.filters[i], activation=None, padding='same')(x)
                x = tfl.BatchNormalization()(x)
                x = tfl.Activation('relu')(x)
                x = tfl.Dropout(self.dropout_rate)(x)

        outputs = tfl.Conv1D(self.num_classes, activation=self.output_activation, padding='same')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)


