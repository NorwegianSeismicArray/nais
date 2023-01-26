import tensorflow as tf

class UnitCircleInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mean=0, stddev=1):
        """Will initialize weights on a hyper circle.
        Usefull for bias initialization for layers where the output is complex.
    
        Args:
            mean (int, optional):  Defaults to 0.
            stddev (int, optional):  Defaults to 1.
        """
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        w = tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
        w, n = tf.linalg.normalize(w, axis=-1)
        return w

    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}