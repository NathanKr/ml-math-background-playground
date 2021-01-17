import numpy as np

x1 = np.array([3 , 13 , 19 , 24 , 29])
print('x1.mean : ',np.mean(x1))
mean_x1 = np.mean(x1)

x2 = np.array([12 , 10 , 29, 33 , 38])
print('x2.mean : ',np.mean(x2))
mean_x2 = np.mean(x2)

num_measurements = x1.size
covariance = np.dot(x1 - mean_x1 , x2 - mean_x2) / (num_measurements - 1)

print('covariance : ' , covariance)