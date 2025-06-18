from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
digits = load_digits()
features = digits.data                    # (1797, 64): 8x8 이미지 벡터
labels = digits.target                    # (1797,): 0~9 클래스 정수

print(features.shape)
print(labels.shape)
print(labels[:30])

# 2. 학습/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 4. 기본 설정
num_features = X_train_std.shape[1] # 64
num_samples = X_train_std.shape[0]  # 1437
num_classes = 10

# 5. 원-핫 인코딩
one_hot = np.eye(num_classes)
y_train_one_hot = one_hot[y_train]
y_test_one_hot = one_hot[y_test]