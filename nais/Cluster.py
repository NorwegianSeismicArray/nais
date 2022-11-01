

import tensorflow as tf

class KMeans(tf.keras.Model):
    def __init__(self, data, num_clusters=10, lr=0.1, name="KMeans"):

        super(KMeans, self).__init__(name=name)
        # Initialalize all the variables
        self.num_clusters = num_clusters
        self.lr = lr
        self.centroids = tf.Variable(tf.keras.initializers.HeNormal()(shape=(num_clusters, data.shape[1])),
                                     name='centroids',
                                     trainable=True)

    def initialize_centroids(self, mean):
        self.centroids.assign(mean)

    def calculate_loss(self, assignments, distances):
        loss = 0
        for c in range(self.num_clusters):
            loss += tf.reduce_mean(tf.where(tf.equal(assignments, c), distances, tf.zeros_like(distances)))

        self.add_loss(loss)

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs, _ = inputs

        distances = tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(inputs, 0), tf.expand_dims(self.centroids,1))), 2)
        assignments = tf.argmin(distances, 0)

        self.calculate_loss(assignments, distances)

        return distances, assignments

