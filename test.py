import numpy as np
N = 3
a = np.random.rand(N,N)
print(a)
b = np.zeros((N,N+1))
print(b)
b[:,:-1] = a
print(a)
print(b)