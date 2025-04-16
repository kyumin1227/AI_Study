import numpy as np

# kin = np.array(1)
# bar = np.array([1, 2, 3])
# foo = np.array([[1], [2], [3]])

# print(f"{kin.shape}, {bar.shape}, {foo.shape}")
# print(f"{type(kin)}, {type(bar)}, {type(foo)}")

# kin = np.zeros((2, 3, 4))
# bar = np.zeros((5, 2))

# print(f"{kin.shape}, {bar.shape}")
# print(f"{kin}, {bar}")

num_features = 3
num_samples = 100

np.random.seed(2)
np.set_printoptions(suppress=True, precision=3)
X =  np.random.rand(num_samples, num_features)
# h(x) = wx1 + wx2 + wx3 + b
w_true = np.random.randint(1, 10, num_features)
b_true = np.random.randn() * 0.5

# y = X[:, 0] * w_true[0] + X[:, 1] * w_true[1] + X[:, 2] * w_true[2] + b_true
y = X @ w_true + b_true
print(X)
print(f"{w_true}\n{b_true}\n\n{y}")

w = np.random.rand(num_features)
b = np.random.randn()

print(w, b)

epochs = 100000
learning_rate = 0.01

for epoch in range(epochs):

    # prediction
    prediction = X @ w + b

    # error
    error = prediction - y

    # gradient
    gradient = X.T @ error / num_samples
    
    # update
    w = w - learning_rate * gradient
    print(w)
    b = b - learning_rate * np.mean(error)
    print(b)

print(w_true)
print(b_true)