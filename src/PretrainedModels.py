

from Models import ImageAutoEncoder, WaveAutoEncoder
import os
from utils import download_weights, extract_weights
from tensorflow import keras

url = 'bitbucket...'

class PretrainedModel(keras.Model):
    def __init__(self,
                 num_classes=10,
                 include_top=True,
                 pooling=None,
                 weights='/models/ImageAutoEncoder_alna_waveforms_depth2_2021-10-05',
                 name='PretrainedModel'):
        super(PretrainedModel, self).__init__(name=name)
        if not os.path.isdir(weights):
            download_weights(url + weights)
            extract_weights(weights + '.zip', 'models/')

        self.autoencoder.load_weights(weights)

        self.model = keras.Sequential()
        self.model.add(self.autoencoder.encoder)

        rank = len(self.autoencoder.encoder.output.shape[1:])

        if pooling == 'avg':
            self.model.add(
                keras.layers.GlobalAveragePooling2D() if rank > 2 else keras.layers.GlobalAveragePooling1D())
        elif pooling == 'max':
            self.model.add(
                keras.layers.GlobalMaxPooling2D() if rank > 2 in weights else keras.layers.GlobalMaxPooling1D())
        else:
            self.model.add(keras.layers.Flatten())

        if include_top:
            for _ in range(3):
                self.model.add(keras.layers.Dense(1024, activation='relu'))
                self.model.add(keras.layers.Dropout(0.2))
            self.model.add(keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

class SpectrogramModel(PretrainedModel):
    def __init__(self,
                 include_top=False,
                 num_classes=10,
                 pooling=None,
                 weights='/models/ImageAutoEncoder_alna_waveforms_depth2_2021-10-05',
                 name='SpectrogramModel'):
        depth = int(weights.split('depth')[-1][0])
        self.autoencoder = ImageAutoEncoder(depth=depth)
        super(SpectrogramModel, self).__init__(name=name,
                                               pooling=pooling,
                                               num_classes=num_classes,
                                               weights=weights,
                                               include_top=include_top)

class WaveformModel(PretrainedModel):
    def __init__(self,
                 include_top=False,
                 num_classes=10,
                 pooling=None,
                 weights='/models/WaveAutoEncoder_alna_waveforms_depth2_2021-10-05',
                 name='WaveformModel'):
        depth = int(weights.split('depth')[-1][0])
        self.autoencoder = WaveAutoEncoder(depth=depth)
        super(WaveformModel, self).__init__(name=name,
                                            pooling=pooling,
                                            num_classes=num_classes,
                                            weights=weights,
                                            include_top=include_top)
