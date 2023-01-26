
import tensorflow as tf 
import numpy as np

class SpectrogramTimeAugment(tf.keras.layers.Layer):
    def __init__(self, prop=0.1, name='SpectrogramTimeAugment'):
        """Augment images in the time domain. 

        Args:
            prop (float, optional): _description_. Defaults to 0.1.
            name (str, optional): _description_. Defaults to 'SpectrogramTimeAugment'.
        """
        super(SpectrogramTimeAugment, self).__init__(name=name)
        self.prop = prop

    def get_config(self):
        return dict(prop=self.prop, name=self.name)

    def call(self, inputs):
        length, _, _ = inputs.shape[1:]
        mask = np.ones(inputs.shape[1:])
        start = np.random.randint(0, int(length * (1 - self.prop)))
        mask[start:start + int(self.prop * length), :, :] = 0.0
        return inputs * np.expand_dims(mask, axis=0)

class SpectrogramFreqAugment(tf.keras.layers.Layer):
    def __init__(self, prop=0.1, name='SpectrogramFreqAugment'):
        """Augment images in the freq domain. 

        Args:
            prop (float, optional): _description_. Defaults to 0.1.
            name (str, optional): _description_. Defaults to 'SpectrogramFreqAugment'.
        """
        super(SpectrogramFreqAugment, self).__init__(name=name)
        self.prop = prop

    def get_config(self):
        return dict(prop=self.prop, name=self.name)

    def call(self, inputs):
        _, height, _ = inputs.shape[1:]
        mask = np.ones(inputs.shape[1:])
        start = np.random.randint(0, int(height * (1 - self.prop)))
        mask[:, start:start + int(self.prop * height), :] = 0.0
        return inputs * np.expand_dims(mask, axis=0)
