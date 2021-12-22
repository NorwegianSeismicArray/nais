import numpy as np

from nais.Datasets import Arces
from nais.Models import AlexNet2D
from nais.Layers import StackedSpectrogram
from nais.Augment import SpectrogramTimeAugment, SpectrogramFreqAugment
import keras_tuner as kt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = Arces()
X, y, info = data.get_X_y(include_info=True, subsample=0.2)
y = LabelEncoder().fit_transform(y)


def create_model():
    inp = tf.keras.layers.Input(X.shape[1:])

    x = StackedSpectrogram()(x)
    x = SpectrogramFreqAugment()(x)
    x = SpectrogramTimeAugment()(x)

    x = AlexNet2D(num_outputs=y.shape[-1], output_type='softmax')(x)

    model = tf.keras.layers.Model(inp,x)
    model.compile('adam','categorical_crossentropy')

    return model

oof = np.zeros(y.shape)

for train_idx, test_idx in StratifiedKFold().split(X,y):

    model = create_model()
    model.fit(
        X[train_idx],y[train_idx],
        validation_data=(X[test_idx],y[test_idx]),
        callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)],
        epochs=10
    )

    oof[test_idx] += model.predict(X[test_idx])

print('OOF accuracy', accuracy_score(y.argmax(axis=-1), oof.argmax(axis=-1)))





