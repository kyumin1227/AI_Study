import numpy as np

# H(x) = 5X + 3X + 4

num_of_samples = 5
num_of_features = 2

# data set
np.random.seed(1)
np.set_printoptions(False, suppress=True)
X = np.random.rand(num_of_samples, num_of_features) * 10
x_true = [5, 3]
b_true = 4
noise = np.random.randn(num_of_samples) * 0.5

print(X)
print(X[0, 1])
print(X[2, 0])
print(X[:, 0])
print(X[:, 1])

y = x_true[0] * X[:, 0] + x_true[1] * X[:, 1] + b_true + noise

print(y)
