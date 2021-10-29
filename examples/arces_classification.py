import numpy as np

from nais.Datasets import Arces
from nais.Models import WaveAlexNet
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

data = Arces()
X, y, info = data.get_X_y(include_info=True, subsample=0.2)

y = LabelEncoder().fit_transform(y)

model = WaveAlexNet(num_outputs=len(np.unique(y)), output_type='multiclass')
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy())
model.fit(X,y, epochs=10, validation_split=0.2, verbose=1)