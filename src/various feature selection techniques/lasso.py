import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
df = pd.read_csv('all_feature_data.csv')

df['categories'] = df['category']
# 불필요한 열 제거
df = df.drop(['Acceleration', 'Lateral acceleration', 'Gas pedal position', 'Brake pedal force', 
              'Gear', 'Steering wheel position', 'Velocity', 'Lateral velocity', 'Vertical velocity', 
              'X axis position', 'Y axis position', 'Z axis position', 'valence', 'arousal', 'dominance', 
              'intensity', 'gender', 'age', 'driving age', 'P-score', 'E-score', 'N-score', 'L-score', 
              'participant', 'category'], axis=1)


# 독립 변수(X)와 종속 변수(y) 분리
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 라벨 인코딩
encoder = LabelEncoder()
y_encoder = encoder.fit_transform(y)
print(y_encoder)

# GridSearchCV용 파라미터 및 KFold 설정
params = {"alpha": [0.00001, 0.001, 0.1]}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 모델 초기화 및 그리드 서치 적용
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X, y_encoder)

# 최적 파라미터 출력
print(f"Best Params: {lasso_cv.best_params_}")

# 최적 파라미터로 Lasso 모델 학습
best_alpha = lasso_cv.best_params_['alpha']
lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(X, y_encoder)

# 피처 중요도 추출 및 절대값 변환
lasso_coef = np.abs(lasso_best.coef_)
print(lasso_coef)

# 피처 중요도 시각화
plt.bar(X.columns, lasso_coef)
plt.xticks(rotation=90)
plt.grid()
plt.title("Feature Selection Based on Lasso")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.ylim(0, 0.05)
plt.show()
