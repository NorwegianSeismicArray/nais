"""
Author: Erik B. Myklebust, erik@norsar.no
2021

Example of use of pretrained models.
"""
from src.utils import spectrogram_minmax_scaler
from src.Models import CreateSpectrogramModel
from src.PretrainedModels import SpectrogramModel
from tensorflow import keras
from sklearn.model_selection import train_test_split
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

model = SpectrogramModel(include_top=True, num_classes=3, weights=weights_path)
for layer in model.layers[:-7]:
    layer.trainable=False

## Add classification task
y = np.random.randint(0,3,size=spectrograms.shape[0]) #For demo purposes.
spectrograms_train, spectrograms_test, y_train, y_test = train_test_split(spectrograms,y,test_size=0.3)

num_classes = len(np.unique(y))
lr = 1e-3

model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.SparseCategoricalCrossentropy())
model.fit(spectrograms_train, y_train, epochs=10)

# Finetune
for layer in model.layers[:-7]:
    layer.trainable=False
model.compile(optimizer=keras.optimizers.Adam(lr/100), loss=keras.losses.SparseCategoricalCrossentropy()) #reduce learning rate for finetuning.
model.fit(spectrograms_train, y_train, epochs=3)
print(model.evaluate(spectrograms_test, y_test))
