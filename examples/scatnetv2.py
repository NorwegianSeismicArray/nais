
import numpy as np
from nais.Models import ScatNetV2

BANKS = (
        {"octaves": 6, "resolution": 6, "quality": 1},
        {"octaves": 9, "resolution": 1, "quality": 3}
    )

network = ScatNetV2(BANKS,
                    bins=1024,
                    sampling_rate=1.0,
                    pool_type='none',
                    combine=True,
                    data_format='channels_last')

X = np.random.normal(size=(128, 1024, 3))
p = network.predict(X)
print(p.shape)
