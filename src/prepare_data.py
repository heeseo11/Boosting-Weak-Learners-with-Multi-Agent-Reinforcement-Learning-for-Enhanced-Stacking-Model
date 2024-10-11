import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 원본 데이터 로드 및 참가자 ID별 데이터 분리
all_data = pd.read_csv('all_feature_data.csv')
split_part = ['P06', 'P08', 'P13', 'P24', 'P32', 'P34', 'P35', 'P36']

# test5에 해당하는 참가자와 train5 참가자 분리
train = all_data[~all_data['participant'].str.contains('|'.join(split_part))]
test = all_data[all_data['participant'].str.contains('|'.join(split_part))]

# 학습 데이터와 테스트 데이터에서 불필요한 열 제거
X_train = train.drop(['category', 'participant'], axis=1)
y_train = train['category']

X_test = test.drop(['category', 'participant'], axis=1)
y_test = test['category']

# y_train, y_test 라벨 인코딩 (한 번만 fit 사용)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성
sgd = SGDClassifier(random_state=42)

sgd.fit(X_train, y_train)
sgd_train_pred = sgd.predict(X_train)
sgd_test_pred = sgd.predict(X_test)

print(classification_report(y_test, sgd_test_pred, digits = 3))

# 튜닝할 파라미터 그리드 정의
param_grid = {
    'loss': ['hinge', 'log'],  # 손실 함수
    'penalty': ['l1', 'l2'],  # 규제 (l1, l2, 혹은 혼합)
    'alpha': [0.001, 0.01, 0.1],  # 규제 강도
}

# GridSearchCV를 사용하여 모델 훈련 및 튜닝
grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 성능 출력
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# 최적의 모델을 얻고 테스트 데이터로 예측
sgd_best_model = grid_search.best_estimator_

sgd_train_tuned_pred = sgd_best_model.predict(X_train)
sgd_test_tuned_pred = sgd_best_model.predict(X_test)

# 분류 보고서 출력
print(classification_report(y_test, sgd_test_tuned_pred, digits = 3))

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_train_pred = knn.predict(X_train)
knn_test_pred = knn.predict(X_test)

print(classification_report(y_test, knn_test_pred))

# 튜닝할 파라미터 그리드 정의
param_grid = {
    'n_neighbors': [3, 5, 7],  # 이웃의 수
    'weights': ['uniform', 'distance'],  # 이웃의 가중치
    'p': [1, 2]  # 거리 측정 방법 (1: 맨하탄 거리, 2: 유클리드 거리)
}

# GridSearchCV를 사용하여 모델 훈련 및 튜닝
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 최적의 모델을 얻고 테스트 데이터로 예측
knn_best_model = grid_search.best_estimator_

knn_train_tuned_pred = knn_best_model.predict(X_train)
knn_test_tuned_pred = knn_best_model.predict(X_test)

# 분류 보고서 출력
print(classification_report(y_test, knn_test_tuned_pred, digits = 3))

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

DT_train_pred = dt.predict(X_train)
DT_test_pred = dt.predict(X_test)

print(classification_report(y_test, DT_test_pred, digits = 3))

# 조정할 파라미터들과 후보값들을 정의합니다.
param_grid = {'max_depth': [1, 2, 3, 4, 5],
              'min_samples_split': [2, 3, 4]}
# GridSearchCV를 이용해 최적의 파라미터를 찾습니다.
grid_dtc = GridSearchCV(dt, param_grid=param_grid, cv=3)
grid_dtc.fit(X_train, y_train)

# 최적의 파라미터 조합으로 학습한 모델로 예측 및 성능 측정
best_dtc = grid_dtc.best_estimator_

DT_train_Tun_pred = best_dtc.predict(X_train)
DT_test_Tun_pred = best_dtc.predict(X_test)

print(classification_report(y_test, DT_test_Tun_pred, digits = 3))

nb = GaussianNB()
nb.fit(X_train, y_train)

# 테스트 데이터로 예측

NB_train_pred = nb.predict(X_train)
NB_test_pred = nb.predict(X_test)

print(classification_report(y_test, NB_test_pred, digits = 3))

param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# 그리드 서치를 통한 파라미터 튜닝
grid_search_NB = GridSearchCV(nb, param_grid=param_grid, cv=3)
grid_search_NB.fit(X_train, y_train)

# 테스트 데이터로 예측
NB_train_Tun_pred = grid_search_NB.predict(X_train)
NB_test_Tun_pred = grid_search_NB.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, NB_test_Tun_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, NB_test_Tun_pred, digits = 3))

lr = LogisticRegression()
lr.fit(X_train, y_train)

# 테스트 데이터로 예측
LR_train_pred = lr.predict(X_train)
LR_test_pred = lr.predict(X_test)

print(classification_report(y_test, LR_test_pred, digits = 3))

# 튜닝할 파라미터 그리드 정의
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization 매개변수
    'C': [0.1, 1, 10],  # 규제 강도 역수
}

# GridSearchCV를 사용하여 모델 훈련 및 튜닝
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 최적의 모델을 얻고 테스트 데이터로 예측
lr_best_model = grid_search.best_estimator_
LR_train_Tun_pred = lr_best_model.predict(X_train)
LR_test_Tun_pred = lr_best_model.predict(X_test)

# 분류 보고서 출력
print(classification_report(y_test, LR_test_Tun_pred, digits=3))

svm = svm.SVC()
svm.fit(X_train, y_train)

# 테스트 데이터로 예측
SVM_train_pred = svm.predict(X_train)
SVM_test_pred = svm.predict(X_test)

print(classification_report(y_test, SVM_test_pred, digits = 3))

param_grid = {
    'C': [10],  # 규제 매개변수
    'kernel': ['linear'],  # 커널 타입
    'gamma': ['scale'],  # 커널 특정 매개변수
}

# GridSearchCV를 사용하여 모델 훈련 및 튜닝
grid_search = GridSearchCV( svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# 최적의 모델을 얻고 테스트 데이터로 예측
svm_best_model = grid_search.best_estimator_
SVM_train_Tun_pred = svm_best_model.predict(X_train)
SVM_test_Tun_pred = svm_best_model.predict(X_test)

# 분류 보고서 출력
print(classification_report(y_test, SVM_test_Tun_pred, digits=3))

# 예측 결과를 데이터프레임으로 정리
train_pred_df = pd.DataFrame({
    'v_sgd': sgd_train_pred, 'sgd': sgd_train_tuned_pred, 
    'v_knn': knn_train_pred, 'knn': knn_train_tuned_pred, 
    'v_dt': DT_train_pred, 'dt': DT_train_Tun_pred,
    'v_nb': NB_train_pred, 'nb': NB_train_Tun_pred,
    'v_lr': LR_train_pred, 'lr': LR_train_Tun_pred,
    'v_svm': SVM_train_pred, 'svm': SVM_train_Tun_pred,
    'pred': y_train, 'participant': train5['participant']
})

test_pred_df = pd.DataFrame({
    'v_sgd': sgd_test_pred, 'sgd': sgd_test_tuned_pred,
    'v_knn': knn_test_pred, 'knn': knn_test_tuned_pred,
    'v_dt': DT_test_pred, 'dt': DT_test_Tun_pred,
    'v_nb': NB_test_pred, 'nb': NB_test_Tun_pred,
    'v_lr': LR_test_pred, 'lr': LR_test_Tun_pred,
    'v_svm': SVM_test_pred, 'svm': SVM_test_Tun_pred,
    'pred': y_test, 'participant': test5['participant']
})

csv_train_file_path = './train_data.csv'
csv_test_file_path = './test_data.csv'

train_pred_df.to_csv(csv_train_file_path, index=False)
test_pred_df.to_csv(csv_test_file_path, index=False)
