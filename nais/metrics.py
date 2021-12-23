
import numpy as np

def _smallestSignedAngleBetween(x, y):
    tau = 2 * np.pi
    a = (x - y) % tau
    b = (y - x) % tau
    return np.where(a < b, -a, b)

def circular_r2_score(true,pred):
    res = sum(_smallestSignedAngleBetween(true,pred)**2)
    tot = sum(_smallestSignedAngleBetween(true,true.mean())**2)
    return 1 - (res/tot)