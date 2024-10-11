import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
import random
import time

# Load data
final_train_data = pd.read_csv('./train_data.csv', index_col=0)
final_test_data = pd.read_csv('./test_data.csv', index_col=0)

# Prepare data
X_train = final_train_data.drop(['category', 'participant'], axis=1)
y_train = final_train_data['category']
X_test = final_test_data.drop(['category', 'participant'], axis=1)
y_test = final_test_data['category']

# Label encoding
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize seed value
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# Initialize decision tree classifier
DT_final = DecisionTreeClassifier(random_state=seed_value)

# Function to compute F1 score and feature importance
def F1_score(input):
    sel_X_train = pd.DataFrame(X_train)[input]
    sel_X_test = pd.DataFrame(X_test)[input]

    scaler = StandardScaler()
    select_X_train = scaler.fit_transform(sel_X_train)
    select_X_test = scaler.transform(sel_X_test)

    DT_final.fit(select_X_train, y_train)
    stack_final = DT_final.predict(select_X_test)

    F1 = f1_score(y_test, stack_final, average='weighted')
    importance = DT_final.feature_importances_
    print(f'Final Meta Model Accuracy: {accuracy_score(y_test, stack_final):.4f}')

    return F1, importance

# Function to get reward based on features
def get_reward(features):
    if len(features) == 0:
        return 0
    f1, feature_importance = F1_score(features)
    R = f1 * 100
    return R, feature_importance

# Function for previous reward
def previous_get_reward(features):
    if len(features) == 0:
        return 0
    f1, feature_importance = F1_score(features)
    R = f1 * 100
    return R, f1, feature_importance

# Initial reward setup
R, previous_F1_score, fi = previous_get_reward([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

epsilon = 0.4
alpha = 0.2
epsilon_decay_rate = 0.995
alpha_decay_rate = 0.995
num_agents = 12

# Initialize variables for Q-learning
Q_values = [[-1, -1] for _ in range(num_agents)]
weight_Array = [...] # 1 - 개별모델성능
current_actions = [0] * num_agents
previous_action = [1] * num_agents
previous_R = previous_F1_score

# Start training loop
start = time.time()

for episode in range(1000):
    print(f"Episode: {episode}")

    for agent in range(num_agents):
        rand_number = random.uniform(0, 1)
        if rand_number > epsilon:
            current_actions[agent] = np.argmax(Q_values[agent])
        else:
            current_actions[agent] = random.choice([0, 1])

    if sum(current_actions) == 0:
        print('All current actions are 0, skipping...')
        pass
    else:
        total_model = [i for i, act in enumerate(current_actions) if act == 1]

        Current_R, feature_importance = get_reward(total_model)
        Improvement_R = Current_R - previous_R

        different_indices = [i for i in range(len(current_actions)) if current_actions[i] != previous_action[i]]
        divide_R = Improvement_R / len(different_indices)

        ratio = [0] * len(weight_Array)
        total_weight = sum(weight_Array[idx] for idx in different_indices)
        for i in different_indices:
            ratio[i] = weight_Array[i] / total_weight

        for agent in different_indices:
            Q_values[agent][current_actions[agent]] += alpha * (divide_R * ratio[agent] - Q_values[agent][current_actions[agent]])

        previous_R = Current_R
        previous_action = current_actions.copy()

        alpha *= alpha_decay_rate
        epsilon *= epsilon_decay_rate

    print(f"Q_values: {Q_values}")

print(f"Training time: {time.time() - start:.2f} seconds")
