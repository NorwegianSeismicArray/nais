


import tensorflow as tf

class PCA(tf.keras.Model):
    def __init__(self, input_shape, n_pca=10, lr=0.1, name="PCA"):

        super(PCA, self).__init__(name=name)
        self.lr = lr
        self.input_shape = input_shape
        self.moving_sigma = tf.Variable(tf.zeros_like((input_shape, n_pca)), name='sigma', trainable=False)

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs, _ = inputs

        singular_values, u, _ = tf.linalg.svd(inputs, full_matrices=False)
        sigma = tf.slice(tf.linalg.diag(singular_values), [0, 0], [self.input_shape, self.n_pca])

        self.moving_sigma.assign(self.moving_pca * self.lr + (1 - self.lr) * sigma)

        pca = tf.linalg.matmul(u, self.moving_sigma)

        return pca


