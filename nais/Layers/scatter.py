import tensorflow as tf 
import numpy as np
from nais.utils import complex_hermite_interp, real_hermite_interp

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
                 learn_filters=True,
                 learn_knots=False,
                 learn_scales=False,
                 hilbert=False):
        """Scattering network layer.

        Computes the convolution modulus and scattering coefficients of the
        input signal.

        Args:
            batch_size (_type_, optional): _description_. Defaults to None.
            j (_type_, optional): _description_. Defaults to None.
            q (_type_, optional): _description_. Defaults to None.
            k (_type_, optional): _description_. Defaults to None.
            pooling_type (str, optional): _description_. Defaults to 'avg'.
            decimation (int, optional): _description_. Defaults to 2.
            pooling (int, optional): _description_. Defaults to 2.
            index (int, optional): _description_. Defaults to 0.
            name (str, optional): _description_. Defaults to 'scattering'.
            learn_filters (bool, optional): _description_. Defaults to True.
            learn_knots (bool, optional): _description_. Defaults to False.
            learn_scales (bool, optional): _description_. Defaults to False.
            hilbert (bool, optional): _description_. Defaults to False.
        """

        super(Scattering, self).__init__(name=name)
        self.decimation = decimation
        self.batch_size = batch_size
        self.learn_filters = learn_filters
        self.learn_knots = learn_knots
        self.learn_scales = learn_scales
        self.hilbert = hilbert
        # Filter bank properties
        self.j = j[index] if type(j) is list else j
        self.q = q[index] if type(q) is list else q
        self.k = k[index] if type(k) is list else k

        if pooling_type == 'avg':
            self.pool = lambda x: tf.nn.avg_pool1d(x, pooling // (decimation ** (index + 1)),
                pooling // (decimation ** (index + 1)), padding='VALID', data_format='NWC')
        elif pooling_type == 'max':
            self.pool = lambda x: tf.nn.max_pool1d(x, pooling // (decimation ** (index + 1)),
                pooling // (decimation ** (index + 1)), padding='VALID', data_format='NWC')
        else:
            self.pool = lambda x: x

    def get_filters(self):
        scales_base = 2 ** (tf.range(self.j * self.q, dtype=tf.float32) / np.float32(self.q))
        scales = scales_base + self.scales_delta

        knots_sum = tf.cumsum(
            tf.clip_by_value(
                tf.expand_dims(self.knots_base, 0) * tf.expand_dims(scales, 1),
                1, self.num_filters - self.k), exclusive=True, axis=1)
        knots = knots_sum - (self.k // 2) * tf.expand_dims(scales, 1)

        # Boundary Conditions
        mask = tf.concat([tf.zeros((1,)), tf.ones((self.k-2,)), tf.zeros((1,))], axis=0)

        if self.hilbert:
            #Centering
            m_null = self.m - mask * tf.math.reduce_mean(self.m, keepdims=True)
            filters = real_hermite_interp(self.time_grid, knots, m_null * tf.squeeze(mask), self.p * mask)

            # Renorm and set filter-bank
            filters_renorm = filters / tf.math.reduce_max(filters, 1, keepdims=True)
            filters_fft = tf.signal.rfft(filters_renorm)
            filters = tf.signal.ifft(tf.concat([filters_fft, tf.zeros_like(filters_fft)], 1))

        else:
            mask = tf.expand_dims(mask, axis=0)
            m_null = self.m - mask * tf.math.reduce_mean(self.m, axis=1, keepdims=True)
            filters = complex_hermite_interp(self.time_grid, knots, m_null * mask, self.p * mask)
            # Renorm and set filter-bank
            filters_renorm = filters / tf.math.reduce_max(filters, 2, keepdims=True)
            filters = tf.complex(filters_renorm[0], filters_renorm[1])

        filters_concat = tf.concat([tf.math.real(filters), tf.math.imag(filters)], 0)
        filters_kernel = tf.expand_dims(tf.transpose(a=filters_concat), 1)
        return filters_kernel

    def build(self, input_shape):
        pi = tf.constant(np.pi)

        extra_octave = 1 if self.learn_scales else 0
        self.num_filters = self.k * 2 ** (self.j + extra_octave)
        time_max = np.float32(self.k * 2 ** (self.j - 1 + extra_octave))
        self.time_grid = tf.linspace(-time_max, time_max, self.num_filters)

        self.scales_delta = tf.Variable(tf.zeros(self.j * self.q), trainable=self.learn_scales, name='scales')
        self.knots_base = tf.Variable(tf.ones(self.k), trainable=self.learn_knots, name='knots')

        if self.hilbert:
            # Create the (real) parameters
            #m = (np.cos(np.arange(self.k) * np.pi) * np.hamming(self.k)).astype(FORMAT)
            #p = (np.zeros(self.k)).astype(FORMAT)

            m = tf.math.cos(tf.range(self.k, dtype='float32') * pi) * tf.signal.hamming_window(self.k)
            p = tf.zeros(self.k)

        else:
            # Create the (complex) parameters
            #m = np.stack([np.cos(np.arange(self.k) * np.pi) * np.hamming(self.k),
            #              np.zeros(self.k) * np.hamming(self.k)]).astype(FORMAT)
            #p = np.stack([np.zeros(self.k),
            #              np.cos(np.arange(self.k) * np.pi) * np.hamming(self.k)]
            #             ).astype(FORMAT)
            m = tf.stack([
                tf.math.cos(tf.range(self.k, dtype='float32') * pi) * tf.signal.hamming_window(self.k),
                tf.zeros(self.k) * tf.signal.hamming_window(self.k)
            ])
            p = tf.stack([
                tf.zeros(self.k),
                tf.math.cos(tf.range(self.k, dtype='float32') * pi) * tf.signal.hamming_window(self.k)
            ])

        self.m = tf.Variable(m, name='m', trainable=self.learn_filters)
        self.p = tf.Variable(p, name='p', trainable=self.learn_filters)

    def call(self, x):
        input_shape = (self.batch_size, *x.shape[1:])

        filters = self.get_filters()
        kernel_size, _, n_filters = filters.shape
        n_filters //= 2

        x_reshaped = tf.reshape(x, [np.prod(input_shape[:-1]), input_shape[-1], 1])
        p = [0, 0], [kernel_size // 2 - 1, kernel_size // 2 + 1], [0, 0]
        x_padded = tf.pad(x_reshaped, p, mode='symmetric')

        x_conv = tf.nn.conv1d(input=x_padded,
                              filters=filters,
                              stride=self.decimation,
                              padding='VALID',
                              data_format='NWC'
                              )

        u = tf.math.sqrt(tf.math.square(x_conv[..., :n_filters]) +
                         tf.math.square(x_conv[..., n_filters:]))

        out_u = tf.reshape(u, (*input_shape[:-1], n_filters, -1))
        out_u = tf.experimental.numpy.swapaxes(out_u, 1, 2)

        pooled = self.pool(u)
        pooled = tf.experimental.numpy.swapaxes(pooled, 1, 2)

        out_s = tf.reshape(pooled, (*input_shape[:-1], self.j * self.q, -1))
        inverse = tf.gradients(ys=x_conv, xs=x, grad_ys=x_conv)[0]
        loss = tf.keras.metrics.mean_squared_error(inverse, tf.stop_gradient(x))
        self.add_loss(tf.math.reduce_mean(loss))

        return out_u, out_s
