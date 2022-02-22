
import numpy as np

def _smallestSignedAngleBetween(x, y):
    """
    Helper function.
    Returns angle between two angles x and y in radians.
    """
    tau = 2 * np.pi
    a = (x - y) % tau
    b = (y - x) % tau
    return np.where(a < b, -a, b)

def circular_r2_score(true,pred):
    """
    Calculates R2 score between angles (in rad).
    Angles are shifted into (0,2pi) where angle differences are calculated.
    Thereafter, same calulation as for normal R2 score.
    params:
        true : np.array
        pred : np.array

    return:
        float : r2 score between the two sets of angles.
    """
    res = sum(_smallestSignedAngleBetween(true,pred)**2)
    tot = sum(_smallestSignedAngleBetween(true,true.mean())**2)
    return 1 - (res/tot)