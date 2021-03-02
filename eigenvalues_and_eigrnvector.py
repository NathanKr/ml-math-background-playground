import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


A = np.array([[1 , 2],[4 , 3]]) 
lamda, v = LA.eig(A)
print(f"eigenvalues\n{lamda}")
print(f"eigenvectors\n{v}")
v1 = v[:,0]
v2 = v[:,1]
print(f"v1\n{v1}")
print(f"v2\n{v2}")

plt.title('eigenvector v1 : blue , eigenvector v2 : green')
plt.plot([0,v1[0]],[0,v1[1]],'blue')
plt.plot([0,v2[0]],[0,v2[1]],'green')
plt.xlim((-1,1))
plt.ylim((-1,1))
plt.grid()
plt.show()

equation1 = np.dot(A,v1) - lamda[0]*v1 # should be 0
print(equation1)
equation2 = np.dot(A,v2) - lamda[1]*v2 # should be 0
print(equation2)

