import numpy as np


def euclidean_distance(x,y):
    """
    Return the euclidean distance of the two vectors.
    """
    return np.sqrt(np.sum(np.power(x-y,2)))


def spherical_distance(x,y, epsilon=1e-5):
    """
    Return the great circle distance of the two vectors.
    """
    norm_x = np.sqrt(np.sum(np.power(x,2)))
    norm_y = np.sqrt(np.sum(np.power(y,2)))
    xx, yy = x/(norm_x+epsilon), y/(norm_y+epsilon)
    return 2*np.pi*np.arccos(np.dot(xx,yy))

