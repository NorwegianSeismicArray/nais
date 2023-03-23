""" 
Various variations of EQTransformer  
Author: Erik 
Email: erik@norsar.no
"""

from nais.Layers import ResnetBlock1D, TransformerBlock
import tensorflow as tf 
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K
import numpy as np

class EarthQuakeTransformer(tf.keras.Model):
    def __init__(self,
                 input_dim,
                 filters=None,
                 kernelsizes=None,
                 resfilters=None,
                 reskernelsizes=None,
                 lstmfilters=None,
                 attention_width=3,
                 dropout=0.0,
                 transformer_sizes=None,
                 kernel_regularizer=None,
                 classify=True,
                 pool_type='max',
                 att_type='additive',
                 activation='relu',
                 name='EarthQuakeTransformer'):
        """
        https://www.nature.com/articles/s41467-020-17591-w

        Example usage:
        import numpy as np
        test = np.random.random(size=(16,1024,3))
        detection = np.random.randint(2, size=(16,1024,1))
        p_arrivals = np.random.randint(2, size=(16,1024,1))
        s_arrivals = np.random.randint(2, size=(16,1024,1))

        model = EarthQuakeTransformer(input_dim=test.shape[1:])
        model.compile(optimizer='adam', loss=['binary_crossentropy',
                                            'binary_crossentropy',
                                            'binary_crossentropy'])

        model.fit(test, (detection,p_arrivals,s_arrivals))

        Args:
            input_dim (tuple): input size of the model.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            resfilters (list, optional): list of number of residual filters. Defaults to None.
            reskernelsizes (list, optional): list of residual filter sizes. Defaults to None.
            lstmfilters (list, optional): list of number of lstm filters. Defaults to None.
            attention_width (int, optional): width of attention mechanism. Defaults to 3. Use None for full. 
            dropout (float, optional): dropout. Defaults to 0.0.
            transformer_sizes (list, optional): list of sizes of attention layers. Defaults to [64, 64].
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            classify (bool, optional): whether to classify phases or provide raw output. Defaults to True.
            att_type (str, optional): attention type. Defaults to 'additive'. 'multiplicative' is also supported. 
            name (str, optional): model name. Defaults to 'EarthQuakeTransformer'.

        """
        super(EarthQuakeTransformer, self).__init__(name=name)

        if filters is None:
            filters = [8, 16, 16, 32, 32, 64, 64]
        if kernelsizes is None:
            kernelsizes = [11, 9, 7, 7, 5, 5, 3]
        invfilters = filters[::-1]
        invkernelsizes = kernelsizes[::-1]
        if resfilters is None:
            resfilters = [64, 64, 64, 64, 64]
        if reskernelsizes is None:
            reskernelsizes = [3, 3, 3, 2, 2]
        if lstmfilters is None:
            lstmfilters = [16, 16]
        if transformer_sizes is None:
            transformer_sizes = [64,64]

        pool_layer = tfl.MaxPooling1D if pool_type == 'max' else tfl.AveragePooling1D

        def conv_block(f,kz):
            return tf.keras.Sequential([tfl.Conv1D(f, kz, padding='same', kernel_regularizer=kernel_regularizer),
                                        tfl.BatchNormalization(),
                                        tfl.Activation(activation),
                                        tfl.Dropout(dropout),
                                        pool_layer(4, strides=2, padding="same")])

        def block_BiLSTM(f, x):
            'Returns LSTM residual block'
            x = tfl.Bidirectional(tfl.LSTM(f, return_sequences=True))(x)
            x = tfl.Conv1D(f, 1, padding='same', kernel_regularizer=kernel_regularizer)(x)
            x = tfl.BatchNormalization()(x)
            return x        

        def _encoder():
            inp = tfl.Input(input_dim)
            def encode(x):
                for f, kz in zip(filters, kernelsizes):
                    x = conv_block(f, kz)(x)
                for f, kz in zip(resfilters, reskernelsizes):
                    x = ResnetBlock1D(f, 
                                      kz, 
                                      activation=activation,
                                      dropout=dropout, 
                                      kernel_regularizer=kernel_regularizer)(x)
                for f in lstmfilters:
                    x = block_BiLSTM(f, x)
                x = tfl.LSTM(f, return_sequences=True, kernel_regularizer=kernel_regularizer)(x)
                for ts in transformer_sizes:
                    x = TransformerBlock(num_heads=8, embed_dim=ts, ff_dim=ts*4, rate=dropout)(x)
                return x
            return tf.keras.Model(inp, encode(inp))

        def inv_conv_block(f,kz):
            return tf.keras.Sequential([tfl.UpSampling1D(2),
                                        tfl.Conv1D(f, kz, padding='same', kernel_regularizer=kernel_regularizer),
                                        tfl.BatchNormalization(),
                                        tfl.Activation(activation),
                                        tfl.Dropout(dropout)])

        def _decoder(input_shape, attention=False, activation='sigmoid', output_name=None):
            inp = tfl.Input(input_shape)
            x = inp
            if attention:
                x = tfl.LSTM(filters[-1], 
                             return_sequences=True, 
                             kernel_regularizer=kernel_regularizer)(x)
                x = TransformerBlock(num_heads=8, ff_dim=filters[-1]*4, embed_dim=filters[-1])(x)

            x = tf.keras.Sequential([inv_conv_block(f, kz) for f, kz in zip(invfilters, invkernelsizes)])(x)
            to_crop = x.shape[1] - input_dim[0]
            of_start, of_end = to_crop//2, to_crop//2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
            if activation is not None:
                x = tfl.Conv1D(1, 1, 
                               padding='same')(x)
                x = tfl.Activation(activation, 
                               name=output_name, 
                               dtype=tf.float32)(x)
            return tf.keras.Model(inp, x)

        self.feature_extractor = _encoder()
        encoded_dim = self.feature_extractor.layers[-1].output.shape[1:]
        
        self.detector = _decoder(encoded_dim, attention=False, activation='sigmoid' if classify else None, output_name='detection')
        self.p_picker = _decoder(encoded_dim, attention=True, activation='sigmoid' if classify else None, output_name='p_phase')
        self.s_picker = _decoder(encoded_dim, attention=True, activation='sigmoid' if classify else None, output_name='s_phase')
        
    @property
    def num_parameters(self):
        s = 0
        for m in [self.feature_extractor, self.detector, self.s_picker, self.p_picker]:
            s += sum([np.prod(K.get_value(w).shape) for w in m.trainable_weights])
        return s

    def call(self, inputs):
        encoded = self.feature_extractor(inputs)
        d = self.detector(encoded)
        p = self.p_picker(encoded)
        s = self.s_picker(encoded)
        return d, p, s


class EarthQuakeTransformerMetadata(EarthQuakeTransformer):
    
    def __init__(self, num_outputs=None, metadata_model=None, eqt_kw=None):
        """Provides a wrapper for EarthQuakeTransformer with a metadata output, eg., when learning back azimuth.

        Args:
            num_outputs (int): Number of numerical metadata to learn. 
            eqt_kw (dict): Args. for EarthQuakeTransformer. 
        """
        super(EarthQuakeTransformerMetadata, self).__init__(**eqt_kw)
        if metadata_model is None:
            self.metadata_model = tf.keras.Sequential([tfl.Flatten(),
                                                   tfl.Dense(128, activation='relu'),
                                                   tfl.Dense(num_outputs)])
        else:
            self.metadata_model = metadata_model

    def call(self, inputs):
        encoded = self.feature_extractor(inputs)
        d = self.detector(encoded)
        p = self.p_picker(encoded)
        s = self.s_picker(encoded)
        m = self.metadata_model(encoded)
        return d, p, s, m
