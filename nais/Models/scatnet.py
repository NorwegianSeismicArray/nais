
""" 
Scattering networks
Author: Erik 
Email: erik@norsar.no
"""

from nais.Mixture import GMM
from nais.Cluster import KMeans
import tensorflow as tf 
import numpy as np

class ScatNet(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 j=[4,6,8],
                 q=[8,2,1],
                 k=7,
                 pooling_type='avg',
                 decimation=2,
                 pooling_size=1024,
                 eps=1e-3,
                 n_pca=5,
                 eps_log=0.001,
                 n_clusters=10,
                 moving_pca=0.9,
                 clustering_method='GMM',
                 name='scatnet',
                 loss_weights=(1.0, 1.0),
                 **filters_kw
                 ):
        """_summary_

        Args:
            input_shape (_type_): _description_
            j (list, optional): _description_. Defaults to [4,6,8].
            q (list, optional): _description_. Defaults to [8,2,1].
            k (int, optional): _description_. Defaults to 7.
            pooling_type (str, optional): _description_. Defaults to 'avg'.
            decimation (int, optional): _description_. Defaults to 2.
            pooling_size (int, optional): _description_. Defaults to 1024.
            eps (_type_, optional): _description_. Defaults to 1e-3.
            n_pca (int, optional): _description_. Defaults to 5.
            eps_log (float, optional): _description_. Defaults to 0.001.
            n_clusters (int, optional): _description_. Defaults to 10.
            moving_pca (float, optional): _description_. Defaults to 0.9.
            clustering_method (str, optional): _description_. Defaults to 'GMM'.
            name (str, optional): _description_. Defaults to 'scatnet'.
            loss_weights (tuple, optional): _description_. Defaults to (1.0, 1.0).
        """
        super(ScatNet, self).__init__(name=name)

        self.batch_size = input_shape[0]
        self.n_pca = n_pca
        self.eps_log = eps_log
        self.moving_pca = moving_pca
        self.n_clusters = n_clusters
        self.eps = eps
        self.loss_weights = loss_weights
        self.clustering_method = clustering_method

        depth = len(j)
        self.ls = [Scattering(batch_size=self.batch_size,
                             index=0,
                             j=j,
                             q=q,
                             k=k,
                             decimation=decimation,
                             pooling=pooling_size,
                             pooling_type=pooling_type,
                             **filters_kw)]

        for i in range(1,depth):
            layer = Scattering(batch_size=self.batch_size,
                               index=i,
                             j=j,
                             q=q,
                             k=k,
                             decimation=decimation,
                             pooling=pooling_size,
                             pooling_type=pooling_type,
                             **filters_kw)

            self.ls.append(layer)

        if pooling_type == 'max':
            self.global_pool = tf.keras.layers.GlobalMaxPooling1D()
        else:
            self.global_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.cluster_loss_tracker = tf.keras.metrics.Mean(name="cluster_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.cluster_loss_tracker,
        ]

    def forward(self, x, training=False):
        def renorm(child, parent, epsilon=1e-3):
            # Extract all shapes.
            batch_size, *_, samples = child.shape
            if epsilon > 0:
                s = tf.experimental.numpy.swapaxes(child,1,2) / (tf.expand_dims(parent, -2) + epsilon)
                batch_size, *_, samples = s.shape
                return tf.reshape(s, [batch_size, -1, samples])
            else:
                return tf.reshape(child, [batch_size, -1, samples])

        x = tf.experimental.numpy.swapaxes(x, 1, 2)
        us, ss = [], []
        for layer in self.ls:
            u, s = layer(x)
            x = u
            us.append(u)
            ss.append(s)

        r = []
        for i in range(1, len(ss)):
            r.append(renorm(ss[i], ss[i-1], self.eps))

        sx = tf.concat(r, axis=1)

        sx = tf.math.log(sx + self.eps_log)
        sx = self.global_pool(sx)

        #sx = tf.reshape(sx, (self.batch_size, -1))
        #sx -= tf.math.reduce_mean(sx, axis=0, keepdims=True)

        return sx

    def predict_step(self, data):
        if type(data) == tuple:
            data, _ = data
        x = data
        proj = self.forward(x, training=False)
        if self.clustering_method == 'GMM':
            sample, prob, mean, logvar = self.clustering(tf.ones((proj.shape[0],1)))
        else:
            dist, assignment = self.clustering(proj)
            prob = tf.nn.softmax(-dist)

        return proj, prob

    def train_step(self, data):
        if type(data) == tuple:
            data, _ = data

        x = data

        with tf.GradientTape() as tape:

            proj = self.forward(x, training=True)
            scat_layers_loss = self.loss_weights[0] * tf.reduce_sum([sum(layer.losses) for layer in self.ls])

            if not hasattr(self, 'clustering'):
                if self.clustering_method == 'GMM':
                    self.clustering = GMM(proj.shape[1], self.n_clusters)
                else:
                    self.clustering = KMeans(proj.shape[1], self.n_clusters)

            if self.clustering_method == 'GMM':
                sample, prob, mean, logvar = self.clustering(tf.ones((proj.shape[0],1)))
                log_likelihood = self.clustering.log_likelihood(proj, prob, mean, logvar) #Should it be proj or sample?
            else:
                dist, _ = self.clustering(proj)
                log_likelihood = tf.reduce_mean(dist)

            cluster_loss = self.loss_weights[1] * log_likelihood
            total_loss = scat_layers_loss + cluster_loss

        w = []
        for layer in self.ls:
            w += layer.variables
        w += self.clustering.variables

        grads = tape.gradient(total_loss, w)
        self.optimizer.apply_gradients(zip(grads, w))
        self.total_loss_tracker.update_state(total_loss)
        self.cluster_loss_tracker.update_state(cluster_loss)
        self.reconstruction_loss_tracker.update_state(scat_layers_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "cluster_loss": self.cluster_loss_tracker.result(),
        }

