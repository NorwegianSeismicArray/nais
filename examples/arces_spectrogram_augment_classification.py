import numpy as np

from nais.Datasets import Arces
from nais.Models import AlexNet2D
from nais.Layers import StackedSpectrogram
from nais.Augment import SpectrogramTimeAugment, SpectrogramFreqAugment
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = Arces()
X, y, info = data.get_X_y(include_info=True, subsample=0.1)

print(X.shape,y.shape)

y = LabelEncoder().fit_transform(y.reshape(-1,1))
num_classes = len(np.unique(y))

def create_model():
    inp = tf.keras.layers.Input(X.shape[1:])

    x = StackedSpectrogram(stack_method='concat', output_dim=(224, 224))(inp)
    x = SpectrogramFreqAugment()(x)
    x = SpectrogramTimeAugment()(x)
    x = AlexNet2D(num_outputs=num_classes, output_type='softmax')(x)

    model = tf.keras.Model(inp,x)
    model.compile('adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

oof = np.zeros(y.shape)

for train_idx, test_idx in StratifiedKFold().split(X,y):

    model = create_model()
    model.summary()
    model.fit(
        X[train_idx],y[train_idx],
        validation_data=(X[test_idx],y[test_idx]),
        callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)],
        epochs=10
    )

    oof[test_idx] += model.predict(X[test_idx])

print('OOF accuracy', accuracy_score(y.argmax(axis=-1), oof.argmax(axis=-1)))


