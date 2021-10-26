### NORSAR Artificial Inteligence System (NAIS, yes, the acronym was choses first)
This repository will eventually contain tools specific to aiding in AI research at NORSAR.

This repository will contain: 
* Baseline models, for waveforms and spectrograms. Possibly pretrained for certain tasks.
* Standard datasets to test models. Development of new models can be difficult. Standard datasets eliminates errors in the data and lets you focus on developing the model. 
  * Classification
  * Regression
  * Masking (eg. arrival picking)
* Augmentation methods.

## Installation
``git clone https://bitbucket:7990/projects/GEOB/repos/nais && python3 nais/setup.py install``

# Quick example

```python
X = ...
y = ...

from nais.Models import AlexNet
model = AlexNet(num_outputs=1) #binary 
model.combile('adam','binary_crossentropy')
model.fit(X,y)
```
