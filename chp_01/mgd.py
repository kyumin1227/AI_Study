import numpy as np


np.random.seed(5)

x = np.array([[1, 2], [3, 4]])  # s1, s2
w = np.array([[2], [3]])    # w1, w2

print(x.shape)
print(w.shape)

print(f"{x}\n{w}\n\n{x@w}")
# x @ w = y^1, y^2