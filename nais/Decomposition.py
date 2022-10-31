


import tensorflow as tf

class PCA(tf.keras.Model):
    def __init__(self, n_pca=10, lr=0.1, name="PCA"):

        super(PCA, self).__init__(name=name)
        self.lr = lr
        self.n_pca = n_pca

    def build(self, input_shape):
        self.moving_sigma = tf.Variable(tf.zeros_like((input_shape[1], self.n_pca)), name='sigma', trainable=False)

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs, _ = inputs

        singular_values, u, _ = tf.linalg.svd(inputs, full_matrices=False)
        sigma = tf.slice(tf.linalg.diag(singular_values), [0, 0], [self.input_shape[1], self.n_pca])

        self.moving_sigma.assign(self.moving_pca * self.lr + (1 - self.lr) * sigma)

        pca = tf.linalg.matmul(u, self.moving_sigma)

        return pca


