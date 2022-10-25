
import numpy as np
from nais.Models import ScatNet

#TESTING
X = np.random.normal(size=(128,1024*16,3))
X_test = np.random.normal(size=(128,1024*16,3))
model = ScatNet(input_shape=(16,*X.shape[1:]), loss_weights=(1e-5,1.0), return_prob=True)
model.compile('adam', loss=None)
model.fit(X, y=None, batch_size=16, verbose=1, epochs=2)
p = model.predict(X_test, batch_size=16, verbose=1)
print(p.shape)
print(p)