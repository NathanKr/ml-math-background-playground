import numpy as np
import math
from utils import variance 

x1 = np.array([3 , 13 , 19 , 24 , 29])
mean_x1 = np.mean(x1)
variance_x1 = variance(x1)
print("estimated population variance variance_x1 : " , variance_x1)

x2 = np.array([12 , 10 , 29, 33 , 38])
mean_x2 = np.mean(x2)
variance_x2 = variance(x2)
print("estimated population variance variance_x2 : " , variance_x2)


num_measurements = x1.size
covariance = np.dot(x1 - mean_x1 , x2 - mean_x2) / (num_measurements - 1)
print("covariance  : ",covariance)


correlation = covariance/(math.sqrt(variance_x1) * math.sqrt(variance_x2) )
print("correlation (1 , -1) : ",correlation)
