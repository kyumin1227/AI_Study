import numpy as np

# 형태 확인

# bar = np.zeros((2))
# foo = np.zeros((3, 2))
# pos = np.zeros((2, 3, 2))

# print(f"bar.shape: {bar.shape}")
# print(f"foo.shape: {foo.shape}")
# print(f"pos.shape: {pos.shape}")

# print(type(np.random.rand(2, 3)))


# 행렬 연산

np.set_printoptions(suppress=True, precision=1)

# bar = np.random.rand(2, 1)

# print(bar)
# print("--" * 10)

# bar *= 10
# print(bar)


# 행렬 연산

# X = np.random.rand(100, 1) * 10
# y = 2.5 * X + np.random.rand(100, 1)

# print(X)
# print("--" * 10)
# print(y)

X = np.random.rand(5, 1) * 10

pos = 2.5 * X
bar = np.random.randn(5, 1) * 2
y = 2.5 * X + bar

print(X)
print("--" * 10)
print(pos)
print("--" * 10)
print(bar)
print("--" * 10)
print(y)
print("--" * 10)
print(y.ravel())    # 고차원 배열을 1차원으로 변환
