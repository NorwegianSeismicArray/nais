import tensorflow as tf
try:
    from kapre import STFT, Magnitude
except ImportError as e:
    print(e)
    print('kapre not installed')

import numpy as np
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K
from nais.utils import complex_hermite_interp, real_hermite_interp
import logging
FORMAT = 'float32'

class StackedSpectrogram(tfl.Layer):
    """
    Creates spectrograms for each channel and stacks them to grayscale.

    output_dim :: tuple
    n_fft :: int
    win_len :: int
    output_dim : tuple, resizing
    num_components : int, number of channels
    stack_method : str, add, mean, concat, None. 
            concat acts on frequency dimension. None concatenates last dim. 
    """
    def __init__(self, n_fft=2048, win_length=128, hop_length=32, output_dim=(64,64), num_components=3, stack_method=None, name='SpectrogramModel'):
        super(StackedSpectrogram, self).__init__(name=name)

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.output_dim = output_dim
        self.num_components = num_components
        self.stack_method = stack_method

    def get_config(self):
        return dict(n_fft = self.n_fft,
                win_length = self.win_length,
                hop_length = self.hop_length,
                output_dim = self.output_dim,
                num_components = self.num_components,
                stack_method = self.stack_method,
                name=self.name)

    def build(self,input_shape):
        inp = tfl.Input(input_shape[1:])
        x = STFT(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)(inp)
        x = Magnitude()(x)
        x = tfl.Lambda(tf.math.square)(x)
        x = tfl.Lambda(lambda x: tf.clip_by_value(x, 1e-11, np.inf))(x)
        x = tfl.Lambda(tf.math.log)(x)
        x = tfl.Lambda(lambda x: tf.split(x, x.shape[-1]//self.num_components, axis=-1))(x)
        if self.stack_method=='add':
            x = tfl.Add()(x)
        elif self.stack_method=='mean':
            x = tfl.Average()(x)
        elif self.stack_method=='concat':
            x = tfl.Concatenate(axis=1)(x)
        else:
            x = tfl.Concatenate(axis=-1)(x)
            
        x = tfl.Resizing(*self.output_dim)(x)
        self.model = tf.keras.Model(inp,x)
        
    def call(self, inputs):
        return self.model(inputs)

class ResidualConv1D(tfl.Layer):
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
    
class Resampling1D(tfl.Layer):
    def __init__(self, length, interpolation="bilinear", **kwargs):
        super(Resampling1D, self).__init__(**kwargs)
        self.length = length
        self.interpolation = interpolation
        
    def build(self, input_shape):
        self.ls = tfl.Resizing(self.length, input_shape[-1], interpolation=self.interpolation)

    def call(self,inputs):
        x = tf.expand_dims(inputs,axis=-1)
        x = self.ls(x)
        return tf.squeeze(x, axis=-1)

    def get_config(self):
        return dict(name=self.name, length=self.length)


class CosSimConv1D(tfl.Layer):
    def __init__(self, units=32, kernel_size=3):
        super(CosSimConv1D, self).__init__()
        self.units = units
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.in_shape = input_shape

        self.flat_size = self.in_shape[1]
        self.channels = self.in_shape[2]

        self.w = self.add_weight(
            shape=(1, self.channels * self.kernel_size, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True)

        self.p = self.add_weight(
            shape=(self.units,), initializer='ones', trainable=True)

        self.q = self.add_weight(
            shape=(1,), initializer='zeros', trainable=True)

    def l2_normal(self, x, axis=None, epsilon=1e-12):
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_inv_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
        return x_inv_norm

    def stack3(self, x):
        x = tf.stack(
            [
                tf.pad(x[:, :-1, :], tf.constant([[0, 0], [1, 0], [0, 0]])),
                x,
                tf.pad(x[:, 1:, :], tf.constant([[0, 0], [0, 1], [0, 0]])),
            ], axis=2)
        return x

    def call(self, inputs, training=None):
        x = self.stack3(inputs)
        x = tf.reshape(x, (-1, self.flat_size, self.channels * self.kernel_size))
        q = tf.square(self.q)
        x_norm = self.l2_normal(x, axis=2) + q
        w_norm = self.l2_normal(self.w, axis=1) + q
        sign = tf.sign(tf.matmul(x, self.w))
        x = tf.matmul(x / x_norm, self.w / w_norm)
        x = tf.abs(x) + 1e-12
        x = tf.pow(x, tf.square(self.p))
        x = sign * x + self.b
        x = tf.reshape(x, (-1, self.in_shape[1], self.units))
        return x


class MaxAbsPool1D(tfl.Layer):
    def __init__(self, pool_size, pad_to_fit=False):
        super(MaxAbsPool1D, self).__init__()
        self.pad = pad_to_fit
        self.pool_size = pool_size

    def compute_output_shape(self, in_shape):
        if self.pad:
            return (in_shape[0],
                    tf.math.ceil(in_shape[1] / self.pool_size),
                    in_shape[2])
        return (in_shape[0],
                (in_shape[1] // self.pool_size),
                in_shape[2])

    def compute_padding(self, in_shape):
        mod_y = in_shape[1] % self.pool_size
        y1 = mod_y // 2
        y2 = mod_y - y1
        self.padding = ((0, 0), (y1, y2), (0, 0))

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_shape = self.compute_output_shape(self.in_shape)
        self.compute_padding(self.in_shape)

    @tf.function
    def stack(self, inputs):
        if self.pad:
            inputs = tf.pad(inputs, self.padding)
        batch_size = tf.shape(inputs)[0]
        max_height = (tf.shape(inputs)[1] // self.pool_size) * self.pool_size
        stack = tf.stack(
            [inputs[:, i:max_height:self.pool_size, :]
             for i in range(self.pool_size)],
            axis=-1)
        return stack

    @tf.function
    def call(self, inputs, training=None):
        stacked = self.stack(inputs)
        inds = tf.argmax(tf.abs(stacked), axis=-1, output_type=tf.int32)
        ks = tf.shape(stacked)
        idx = tf.stack([
            *tf.meshgrid(
                tf.range(0, ks[0]),
                tf.range(0, ks[1]),
                tf.range(0, ks[2]),
                indexing='ij'
            ), inds],
            axis=-1)

        x = tf.gather_nd(stacked, idx)
        x = tf.reshape(x, (-1, *self.out_shape[1:]))
        return x


class ResnetBlock1D(tfl.Layer):
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


class SeqSelfAttention(tfl.Layer):
    """Layer initialization. modified from https://github.com/CyberZHG
    For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
    :param units: The dimension of the vectors that used to calculate the attention weights.
    :param attention_width: The width of local attention.
    :param attention_type: 'additive' or 'multiplicative'.
    :param return_attention: Whether to return the attention weights for visualization.
    :param history_only: Only use historical pieces of data.
    :param kernel_initializer: The initializer for weight matrices.
    :param bias_initializer: The initializer for biases.
    :param kernel_regularizer: The regularization for weight matrices.
    :param bias_regularizer: The regularization for biases.
    :param kernel_constraint: The constraint for weight matrices.
    :param bias_constraint: The constraint for biases.
    :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                              in additive mode.
    :param use_attention_bias: Whether to use bias while calculating the weights of attention.
    :param attention_activation: The activation used for calculating the weights of attention.
    :param attention_regularizer_weight: The weights of attention regularizer.
    :param kwargs: Parameters for parent class.
    """

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.attention_activation = tf.keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = tf.keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': tf.keras.regularizers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.regularizers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'attention_activation': tf.keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e = e * K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx())
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        input_len = inputs.get_shape().as_list()[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}


class FeedForward(tfl.Layer):
    """Position-wise feed-forward layer. modified from https://github.com/CyberZHG
    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y


class Scattering(tf.keras.layers.Layer):
    """Learnable scattering network layer."""

    def __init__(self,
                 batch_size=None,
                 j=None,
                 q=None,
                 k=None,
                 pooling_type='avg',
                 decimation=2,
                 pooling=2,
                 index=0,
                 name='scattering',
                 **filters_kw):
        """Scattering network layer.

        Computes the convolution modulus and scattering coefficients of the
        input signal.

        Arguments
        ---------
            x: :class:`~tensorflow.Tensor()`
                Input data of shape ``(batch_size, channels, patch_shape).
        """
        super(Scattering, self).__init__(name=name)
        self.decimation = decimation
        self.batch_size = batch_size

        # Filter bank properties
        self.j = j = j[index] if type(j) is list else j
        self.q = q = q[index] if type(q) is list else q
        self.k = k = k[index] if type(k) is list else k

        # Initialize filter bank (n_features, n_samples), then concatenate the
        # real and imaginary parts to do a single convolution with stacked
        # filters (2 * n_features, n_samples) and reshape filter bank with the
        # same logical dimensions of input data (n_samples, 1, 2 * n_features)
        filters = self.init_filters(j, q, k, **filters_kw)
        self.n_filters, self.kernel_size = filters.get_shape().as_list()
        filters_concat = tf.concat([tf.math.real(filters), tf.math.imag(filters)], 0)
        self.filters_kernel = tf.expand_dims(tf.transpose(a=filters_concat), 1)

        # Differentiate the case of one input channel or multiple
        # which needs reshaping in order to treat them independently

        # Scattering pooling setup
        if pooling_type == 'avg':
            self.pool = lambda x: tf.nn.avg_pool1d(x, pooling // (decimation ** (index + 1)),
                pooling // (decimation ** (index + 1)), padding='VALID', data_format='NWC')
        elif pooling_type == 'max':
            self.pool = lambda x: tf.nn.max_pool1d(x, pooling // (decimation ** (index + 1)),
                pooling // (decimation ** (index + 1)), padding='VALID', data_format='NWC')
        else:
            self.pool = lambda x: x

    def call(self, x):
        input_shape = (self.batch_size, *x.shape[1:])

        x_reshaped = tf.reshape(x, [np.prod(input_shape[:-1]), input_shape[-1], 1])
        p = [0, 0], [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1], [0, 0]
        x_padded = tf.pad(x_reshaped, p, mode='symmetric')

        x_conv = tf.nn.conv1d(input=x_padded,
                              filters=self.filters_kernel,
                              stride=self.decimation,
                              padding='VALID',
                              data_format='NWC'
                              )

        u = tf.math.sqrt(tf.math.square(x_conv[..., :self.n_filters]) +
                         tf.math.square(x_conv[..., self.n_filters:]))

        out_u = tf.reshape(u, (*input_shape[:-1], self.n_filters, -1))
        out_u = tf.experimental.numpy.swapaxes(out_u, 1, 2)

        pooled = self.pool(u)
        pooled = tf.experimental.numpy.swapaxes(pooled, 1, 2)

        out_s = tf.reshape(pooled, (*input_shape[:-1], self.j * self.q, -1))
        inverse = tf.gradients(ys=x_conv, xs=x, grad_ys=x_conv)[0]

        self.add_loss(tf.nn.l2_loss(inverse - tf.stop_gradient(x)) / np.prod(input_shape))

        return out_u, out_s

    def init_filters(self, j, q, k,
                     learn_scales=False,
                     learn_knots=False,
                     learn_filters=True,
                     hilbert=False):
        """Create the filter bank."""
        # If the scales are learnable, allows to go toward lower frequencies.
        extra_octave = 1 if learn_scales else 0
        self.filter_samples = k * 2 ** (j + extra_octave)

        # Define the time grid onto integers such as
        # [-k * 2 ** (j - 1), ..., -1, 0, 1, ..., k * 2 ** (j - 1)]
        # We change the range depending on if the extra octave was added
        time_max = np.float32(k * 2**(j - 1 + extra_octave))
        time_grid = tf.linspace(-time_max, time_max, self.filter_samples)

        # Scales
        # ------
        # The first scale is at the Nyquist frequency, the increasing scales
        # go to lower frequencies.
        # Note: the following method might not be computationally optimal but
        # is the most stable for learning and precision.
        scales_base = 2**(tf.range(j * q, dtype=tf.float32) / np.float32(q))
        scales_delta = tf.Variable(
            tf.zeros(j * q), trainable=learn_scales, name='scales')
        scales = scales_base + scales_delta

        # Now ensure that the scales are strictly increasing in case of
        # delta scales are too high amplitude. If not, shift by the right
        # amount get the correcting shifts being the cases with negative
        # increase of the scales and the nyquist offset which ensures that the
        # smallest filter has scale of at least 1.
        nyquist_offset = scales + \
            tf.stop_gradient(tf.one_hot(0, j * q) * tf.nn.relu(1 - scales[0]))
        scales_correction = tf.concat(
            [tf.zeros(1),
             tf.nn.relu(nyquist_offset[:-1] - nyquist_offset[1:])], 0)
        self.scales = nyquist_offset + \
            tf.stop_gradient(tf.cumsum(scales_correction))

        # The knots are defined at each scale. We start from the Nyquist
        # frequency where the knots must be [-k//2, ..., -1, 0, 1, ..., k//2]
        knots_base = tf.Variable(tf.ones(k), trainable=learn_knots, name='knots')

        # We compute the scaled differences first in order to clip to ensure
        # that you can not have (at any scale) as difference of less than 1,
        # corresponding to a Nyquist node. We then cumsum along the time axis
        # to get the scaled positions which are not yet starting at the
        # correct index they start for now at 0. This step is crucial to
        # ensure that during learning, there is no degenracy and aliasing
        # if trying to put knots closer together than 1. This is also crucial
        # for the high frequency filters as the nyquist one will always violate
        # the sampling if the knots are contracted.
        knots_sum = tf.cumsum(
            tf.clip_by_value(
                tf.expand_dims(knots_base, 0) * tf.expand_dims(self.scales, 1),
                1, self.filter_samples - k), exclusive=True, axis=1)
        self.knots = knots_sum - (k // 2) * tf.expand_dims(self.scales, 1)

        # Interpolation init, add the boundary condition mask and remove the
        # mean filters of even indices are the real parts and odd indices are
        # imaginary part
        if hilbert is True:

            # Create the (real) parameters
            m = (np.cos(np.arange(k) * np.pi) * np.hamming(k)).astype(FORMAT)
            p = (np.zeros(k)).astype(FORMAT)
            self.m = tf.Variable(m, name='m', trainable=learn_filters)
            self.p = tf.Variable(p, name='p', trainable=learn_filters)

            # Boundary Conditions and centering
            mask = np.ones(k, dtype=np.float32)
            mask[0], mask[-1] = 0, 0
            m_null = self.m - tf.reduce_mean(input_tensor=self.m[1:-1])
            filters = real_hermite_interp(
                time_grid, self.knots, m_null * mask, self.p * mask)

            # Renorm and set filter-bank
            filters_renorm = filters / tf.reduce_max(input_tensor=filters, axis=1, keepdims=True)
            filters_fft = tf.signal.rfft(filters_renorm)
            filters = tf.signal.ifft(
                tf.concat([filters_fft, tf.zeros_like(filters_fft)], 1))

        else:
            # Create the (complex) parameters
            m = np.stack([np.cos(np.arange(k) * np.pi) * np.hamming(k),
                          np.zeros(k) * np.hamming(k)]).astype(FORMAT)
            p = np.stack([np.zeros(k),
                          np.cos(np.arange(k) * np.pi) * np.hamming(k)]
                         ).astype(FORMAT)
            self.m = tf.Variable(m, name='m', trainable=learn_filters)
            self.p = tf.Variable(p, name='p', trainable=learn_filters)

            # Boundary Conditions and centering
            mask = np.ones((1, k), dtype=np.float32)
            mask[0, 0], mask[0, -1] = 0, 0
            m_null = self.m - \
                tf.reduce_mean(input_tensor=self.m[:, 1:-1], axis=1, keepdims=True)
            filters = complex_hermite_interp(
                time_grid, self.knots, m_null * mask, self.p * mask)

            # Renorm and set filter-bank
            filters_renorm = filters / tf.reduce_max(input_tensor=filters, axis=2, keepdims=True)
            filters = tf.complex(filters_renorm[0], filters_renorm[1])

        # Define the parameters for saving
        #self.variables = self.m, self.p, self.scales, self.knots
        return filters


