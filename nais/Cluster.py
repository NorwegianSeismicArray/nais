

import tensorflow as tf

class KMeans(tf.keras.Model):
    def __init__(self, o_shape, num_clusters=10, lr=0.1, name="KMeans"):

        super(KMeans, self).__init__(name=name)
        # Initialalize all the variables
        self.num_clusters = num_clusters
        self.lr = lr
        w_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.centroids = tf.Variable(
            initial_value=w_init(shape=(num_clusters, o_shape), dtype="float32"),
            name='mean_f',
            trainable=True)

    def initialize_centroids(self, mean):
        self.centroids.assign(mean)

    def update_centroids(self, inputs, distances, assignments):
        means = []
        for c in range(self.num_clusters):
            means.append(tf.reduce_mean(tf.gather(inputs, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1]))

        new_centroids = tf.concat(means,0)
        self.centroids.assign(self.centroids * (1-self.lr) + new_centroids * self.lr)

    def call(self, inputs):

        distances = tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(inputs, 0), tf.expand_dims(self.centroids,1))), 2)
        assignments = tf.argmin(distances, 0)

        return distances, assignments

