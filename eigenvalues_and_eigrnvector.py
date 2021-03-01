import numpy as np
from numpy import linalg as LA


A = np.array([[1 , 2],[4 , 3]]) 
w, v = LA.eig(A)
print(f"eigenvalues\n{w}")
print(f"eigenvectors\n{v}")

# equation = np.dot(A,v) - np.dot(lamda,v)
# print(equation)