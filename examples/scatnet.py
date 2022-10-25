
import numpy as np
from nais.Models import ScatNet

#TESTING
X = np.random.normal(size=(128,1024*16,1))
X_test = np.random.normal(size=(128,1024*16,1))
model = ScatNet(input_shape=(16,*X.shape[1:]))
model.compile('adam', loss=None)
model.fit(X, y=None, batch_size=16, verbose=1, epochs=2)
p = model.predict(X_test, batch_size=16, verbose=1)
print(p.shape)
print(p)