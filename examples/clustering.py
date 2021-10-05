"""
Author: Erik B. Myklebust, erik@norsar.no
2021

Example of use of pretrained models.
"""
from src.utils import spectrogram_minmax_scaler
from src.PretrainedModels import SpectrogramModel
from src.Models import CreateSpectrogramModel
from sklearn.cluster import OPTICS
import numpy as np

dataset = 'alna_waveforms'

X = np.load(f'data/{dataset}.npy')
weights_path = f'models/AutoEncoder-depth2-{dataset}-2021-10-05'

X = X[:10]

# Seperate stations into three components.
X = np.concatenate([X[:,:,3*i:3*(i+1)] for i in range(X.shape[-1]//3)],axis=0)

#Use this or spectrograms needs to have shape = (256,256,3) and be log-normalized.
csm = CreateSpectrogramModel(n_fft=512, win_length=128, hop_length=32)

# Create spectrograms and normalize.
spectrograms = csm.predict(X, verbose=1)
spectrograms = spectrogram_minmax_scaler(spectrograms)

model = SpectrogramModel(include_top=False, weights=weights_path)

encoded_features = model.predict(spectrograms)

## Clustering
clm = OPTICS()
p = clm.fit_predict(encoded_features)
print(np.unique(p, return_counts=True))
