import numpy as np

m1 = np.array([[1,2,3],[4,5,6]])
m2 = np.array([[1,2,3],[2,4,6]])
m3 = np.array([[0,0,0],[0,0,0]])

print(f"rank of m1 : {np.linalg.matrix_rank(m1)}")
print(f"rank of m2 : {np.linalg.matrix_rank(m2)}")
print(f"rank of m3 : {np.linalg.matrix_rank(m3)}")

