
import numpy as np

A = np.array([[1 , 2],[4 , 3]]) 
u, s, vh = np.linalg.svd(A)

print(f"u : \n{u}")
print(f"s : \n{s}")
print(f"vh : \n{vh}")
s_mat = np.diag(s)
print(f"s_mat : \n{s_mat}")
print(f"u * s * vh\n",np.dot(np.dot(u,s_mat),vh))