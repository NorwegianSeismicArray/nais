
import tensorflow as tf 
import tensorflow.keras.layers as tfl 
from nais.Layers import ResidualConv1D

class WaveNet(tf.keras.Model):

    def __init__(self, 
                 num_outputs=None, 
                 kernel_size=3, 
                 output_type='binary', 
                 pooling=None, 
                 filters=None, 
                 stacked_layers=None, 
                 name='WaveNet'):
        """https://deepmind.com/blog/article/wavenet-generative-model-raw-audio

        Args:
            num_outputs (int, optional): Number of outputs of model. Defaults to None.
            kernel_size (int, optional): list of kernel sizes. Defaults to 3.
            output_type (str, optional): problem type, 'binary', 'multiclass', 'multilabel' supported. Defaults to 'binary'.
            pooling (str, optional): type of pooling to apply. Defaults to None.
            filters (list, optional): list of number of filters. Defaults to None.
            stacked_layers (list, optional): number of stacked layers. Defaults to None.
            name (str, optional): model name. Defaults to 'WaveNet'.

        Raises:
            NotImplementedError: If pooling is not in 'avg', 'max', or None. 
        """
        super(WaveNet, self).__init__(name=name)

        if filters is None:
            self.filters = 16
        else:
            self.filters = filters

        if stacked_layers is None:
            self.stacked_layers = [12,8,4,1]
        else:
            self.stacked_layers = stacked_layers

        self.ls = []
        for i,sl in enumerate(self.stacked_layers):
            self.ls.append(tfl.Conv1D(self.filters*(i+1), 1, padding='same'))
            self.ls.append(ResidualConv1D(self.filters*(i+1), kernel_size, sl))
		
        if pooling is None:
            self.ls.append(tfl.Flatten())
        elif pooling == 'avg':
            self.ls.append(tfl.GlobalAveragePooling1D())
        elif pooling == 'max':
            self.ls.append(tfl.GlobalMaxPooling1D())
        else:
            raise NotImplementedError(pooling + 'no implemented')
        
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
