
import tensorflow as tf 
import tensorflow.keras.layers as tfl 

class AlexNet2D(tf.keras.Model):
    def __init__(self, 
                 kernel_sizes=None, 
                 num_outputs=None, 
                 output_type='binary', 
                 pooling='max', 
                 name='AlexNet2D'):
        """https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98

        Args:
            kernel_sizes (list, optional): list of kernel sizes. Defaults to None.
            filters (list, optional): list of number of filters. Defaults to None.
            num_outputs (int, optional): number of outputs. Defaults to None.
            output_type (str, optional): problem type, 'multiclass', 'multilabel'. Defaults to 'binary'.
            pooling (str, optional): pooling type. Defaults to 'max'.
            name (str, optional): model name. Defaults to 'AlexNet2D'.
        """
        
        super(AlexNet2D, self).__init__(name=name)
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        assert len(kernel_sizes) == 5
        assert pooling in [None, 'max', 'avg']

        if pooling == 'max':
            pooling_layer = tfl.MaxPooling2D
        elif pooling == 'avg':
            pooling_layer = tfl.AveragePooling2D
        else:
            pooling_layer = lambda **kwargs: tfl.Activation('linear')

        self.ls = [
            tfl.Conv2D(filters=96, kernel_size=kernel_sizes[0], strides=(4, 4), activation='relu',
                                   padding='same'),
            tfl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tfl.Conv2D(filters=256, kernel_size=kernel_sizes[1], strides=(1, 1), activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=(3, 3), strides=(2, 2)),
            tfl.Conv2D(filters=384, kernel_size=kernel_sizes[2], strides=(1, 1), activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=384, kernel_size=kernel_sizes[3], strides=(1, 1), activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=256, kernel_size=kernel_sizes[4], strides=(1, 1), activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=(2, 2)),
            tfl.Flatten(),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
        ]

        if num_outputs is not None:
            if output_type == 'binary':
                assert num_outputs == 1
                act = 'sigmoid'
            elif output_type == 'multiclass':
                assert num_outputs > 1
                act = 'softmax'
            elif output_type == 'multilabel':
                assert num_outputs > 1
                act = 'sigmoid'
            else:
                act = 'linear'

            self.ls.append(tfl.Dense(num_outputs, activation=act))

    def call(self, inputs):
        x = inputs
        for layer in self.ls:
            x = layer(x)
        return x


class AlexNet1D(tf.keras.Model):
    def __init__(self, 
                 kernel_sizes=None, 
                 filters=None, 
                 num_outputs=None, 
                 output_type='binary', 
                 pooling='max',
                 name='AlexNet1D'):
        """1D AlexNet

        Args:
            kernel_sizes (list, optional): list of kernel sizes. Defaults to None.
            filters (list, optional): list of number of filters. Defaults to None.
            num_outputs (int, optional): number of outputs. Defaults to None.
            output_type (str, optional): problem type, 'multiclass', 'multilabel'. Defaults to 'binary'.
            pooling (str, optional): pooling type. Defaults to 'max'.
            name (str, optional): model name. Defaults to 'AlexNet1D'.
        """
        super(AlexNet1D, self).__init__(name=name)
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        if filters is None:
            filters = [96, 256, 384, 384, 256]
        
        assert len(kernel_sizes) == 5
        assert pooling in [None, 'none', 'max', 'avg']

        if pooling == 'max':
            pooling_layer = tfl.MaxPooling1D
        elif pooling == 'avg':
            pooling_layer = tfl.AveragePooling1D
        else:
            pooling_layer = lambda **kwargs: tfl.Activation('linear')

        self.ls = [
            tfl.Conv1D(filters=filters[0], kernel_size=kernel_sizes[0], strides=4, activation='relu',
                                   padding='same'),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Conv1D(filters=filters[1], kernel_size=kernel_sizes[1], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Conv1D(filters=filters[2], kernel_size=kernel_sizes[2], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv1D(filters=filters[3], kernel_size=kernel_sizes[3], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv1D(filters=filters[4], kernel_size=kernel_sizes[4], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Flatten(),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
        ]

        if num_outputs is not None:
            if output_type == 'binary':
                assert num_outputs == 1
                act = 'sigmoid'
            elif output_type == 'multiclass':
                assert num_outputs > 1
                act = 'softmax'
            elif output_type == 'multilabel':
                assert num_outputs > 1
                act = 'sigmoid'
            else:
                act = 'linear'

            self.ls.append(tfl.Dense(num_outputs, activation=act))

    def call(self, inputs):
        x = inputs
        for layer in self.ls:
            x = layer(x)
        return x
