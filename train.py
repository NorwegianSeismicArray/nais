"""
Author: Erik B. Myklebust, erik@norsar.no
2021

This file trains the autoencoder at different depths using different datasets.

Datasets:
 - GeobyIT. 6 sensors at Alna. 5 days of 40 second windows.
"""

MAX_EPOCHS = 200

import numpy as np
from src.NModels import SpectrogramModel, ImageAutoEncoder, WaveAutoEncoder
from tensorflow import keras
from datetime import date
from src.utils import spectrogram_minmax_scaler, waveform_minmax_scaler

from tslearn.preprocessing import TimeSeriesResampler

dataset = 'alna_waveforms'

X = np.load(f'data/{dataset}.npy')

# Seperate stations into three components.
X = np.concatenate([X[:,:,3*i:3*(i+1)] for i in range(X.shape[-1]//3)],axis=0)

for model in [ImageAutoEncoder, WaveAutoEncoder]:
    if 'Image' in str(model):
        spectrogram_model = SpectrogramModel(n_fft=512, win_length=128, hop_length=32)
        # Create spectrograms and normalize.
        x = spectrogram_model.predict(X, verbose=1)
        x = spectrogram_minmax_scaler(x)
    else:
        x = waveform_minmax_scaler(X)
        x = TimeSeriesResampler(4096).fit_transform(x)

    for d in range(2, 5):
        autoencoder = model(depth=d)

        autoencoder.compile(optimizer='adam',
                            loss='mse')
        autoencoder.fit(x,
                        x,
                        epochs=MAX_EPOCHS,
                        batch_size=64,
                        callbacks=[keras.callbacks.EarlyStopping('loss', patience=10)])

        autoencoder.save_weights(f'models/{str(autoencoder)}-{dataset}-{date.today()}', save_format='tf')

