from sklearn.model_selection import train_test_split
import numpy as np

# X : 입력 값
X = np.random.rand(10, 2) * 5
# y : 출력 값 (정답)
y = np.random.randint(0, 2, size=10)

# random_state = 시드 값으로 같은 값이면 같은 실행이 보장됨
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)   # 4개의 파라미터는 필수

print(f"X_train.shape: {X_train.shape}")
print(X_train)

print(f"y_train.shape: {y_train.shape}")
print(y_train)
