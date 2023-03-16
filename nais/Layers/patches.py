import tensorflow as tf
import tensorflow.keras.layers as tfl

class Patches1D(tfl.Layer):
    def __init__(self, patch_size, patch_stride):
        super(Patches1D, self).__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        images = tf.expand_dims(images, axis=-1)
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_stride, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        
        return patches