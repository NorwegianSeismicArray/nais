"""
Author: Erik B. Myklebust, erik@norsar.no
2021

Example of use of pretrained models.
"""
from src.utils import spectrogram_minmax_scaler
from src.NModels import AutoEncoder, SpectrogramModel
from sklearn.cluster import OPTICS
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

dataset = 'alna_waveforms'

X = np.load(f'data/{dataset}.npy')
weights_path = f'models/AutoEncoder-depth2-{dataset}-2021-10-05'

X = X[:10]

# Seperate stations into three components.
X = np.concatenate([X[:,:,3*i:3*(i+1)] for i in range(X.shape[-1]//3)],axis=0)

#Use this or spectrograms needs to have shape = (256,256,3)
spectrogram_model = SpectrogramModel(n_fft=512, win_length=128, hop_length=32)

# Create spectrograms and normalize.
spectrograms = spectrogram_model.predict(X, verbose=1)
spectrograms = spectrogram_minmax_scaler(spectrograms)

autoencoder = AutoEncoder(depth=2)
autoencoder.load_weights(weights_path)
encoder = autoencoder.encoder

encoded_features = encoder.predict(spectrograms)

## Clustering
clm = OPTICS()
p = clm.fit_predict(encoded_features)
print(np.unique(p, return_counts=True))
