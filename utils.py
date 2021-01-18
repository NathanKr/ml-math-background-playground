import numpy as np
import math

# this is the estimated population variance where the denominator is n-1
def variance(x):
    return np.var(x, ddof=1)

def covariance(x1,x2):
    mean_x1 = np.mean(x1)
    mean_x2 = np.mean(x2)
    num_measurements = x1.size # should be same as x2.size
    return np.dot(x1 - mean_x1 , x2 - mean_x2) / (num_measurements - 1)

# ---- correlation is always between -1,1
def correlation(x1,x2):
    variance_x1 = variance(x1)
    variance_x2 = variance(x2)
    return covariance/(math.sqrt(variance_x1) * math.sqrt(variance_x2) )
