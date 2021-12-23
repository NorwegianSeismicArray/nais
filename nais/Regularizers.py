
import tensorflow as tf

class UnitNormRegularizer(tf.keras.regularizers.Regularizer):
    """
    Penalize outputs not on the unit hyper circle.
    Usefull when output is complex.
    """

    def __init__(self, penalty=0.):
        self.penalty = penalty

    def __call__(self, x):
        x, n = tf.linalg.normalize(x, axis=-1)
        return self.penalty * tf.reduce_sum(tf.math.square(1 - n))

    def get_config(self):
        return {'penalty', self.penalty}