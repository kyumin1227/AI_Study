import numpy as np

NUM_FEATURES = 4
NUM_SAMPLES = 1000
EPOCHS = 100000

np.random.seed(5)

# 데이터 세팅
X = np.random.rand(NUM_SAMPLES, NUM_FEATURES)
w_true = np.random.randint(1, 11, (NUM_FEATURES, 1))
b_true = np.random.randn() * 0.5

y = X @ w_true + b_true

# mgd 계산
w = np.random.rand(NUM_FEATURES, 1)
b = np.random.rand()

learning_rate = 0.01

for _ in range(EPOCHS):

    # 예측값 계산
    predict_y = X @ w + b

    # 오차
    error = predict_y - y

    # 기울기 (전치 작업 진행 후 error와 행렬 곱)
    gradient_w = X.T @ error / NUM_SAMPLES
    gradient_b = error.mean()

    # w, b 업데이트
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

print(f"W_true: {w_true}\nB_true: {b_true}")
print(f"W: {w}\nB: {b}")