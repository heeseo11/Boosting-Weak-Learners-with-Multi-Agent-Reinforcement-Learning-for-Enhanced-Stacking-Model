import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('all_feature_data.csv')

df['categories'] = df['category']
df = df.drop(['Acceleration', 'Lateral acceleration', 'Gas pedal position',
       'Brake pedal force', 'Gear', 'Steering wheel position', 'Velocity',
       'Lateral velocity', 'Vertical velocity', 'X axis position',
       'Y axis position', 'Z axis position', 'valence', 'arousal', 'dominance',
       'intensity', 'gender', 'age', 'driving age', 'P-score', 'E-score',
       'N-score', 'L-score', 'participant', 'category'], axis = 1)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

encoder = LabelEncoder()
encoder.fit(y)
y_encoder = encoder.transform(y)
print(y_encoder)

gbdt = GradientBoostingClassifier()

gbdt.fit(X, y_encoder)

# 특성 중요도 확인
feature_importances = gbdt.feature_importances_
print("Feature Importances:", feature_importances)
names=X.columns

# plotting the Column Names and Importance of Columns. 
plt.bar(names, feature_importances)
plt.xticks(rotation=90)
plt.grid()
plt.title("Feature Selection Based on GBDT")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.ylim(0, 0.3)
plt.show()
