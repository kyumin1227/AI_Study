from sklearn.linear_model import SGDRegressor
import numpy as np

np.set_printoptions(suppress=True, precision=1)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(3, 1)
y = y.ravel()

# 모델 생성 후 하이퍼 파라미터 설정
model = SGDRegressor(
    max_iter=100,   # 에폭
    learning_rate="constant",   # 학습률 설정
    eta0=0.001, # 학습률 초기값
    penalty=None,   # 정규화 제거
    random_state=0  # 시드
)

# 학습 실시
model.fit(X, y)
