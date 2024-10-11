# Boosting Weak Learners with Multi-Agent Reinforcement Learning for Enhanced Stacking Models: An Application on Driver Emotion Classification

paper cite 넣기

## 🔑 Keywords

![Multi-Agent Reinforcement Learning](https://img.shields.io/badge/Multi--Agent%20Reinforcement%20Learning-3776AB?style=flat&logo=python&logoColor=white)
![Stacking Model](https://img.shields.io/badge/Stacking%20Model-276DC3?style=flat&logo=stackexchange&logoColor=white)
![Ensemble Model](https://img.shields.io/badge/Ensemble%20Model-FF6F00?style=flat&logo=google-scholar&logoColor=white)
![Driver Emotion Detection](https://img.shields.io/badge/Driver%20Emotion%20Detection-EE4C2C?style=flat&logo=dribbble&logoColor=white)

<img width="500" alt="image" src="https://github.com/user-attachments/assets/716f20db-c0c6-4f97-a792-86bb75f89eb7">
<p>Fig. 1. Architecture of the Proposed Stacking Model using Multi-Agent Reinforcement Learning</p>

## Agent Definition
Agents are defined as $AG_{n}$, with each agent representing one of the 12 individual models in the stacking model.

## Action Definition
Each agent in the stacking model has two actions for model selection: selecting a model (1) and not selecting a model (0). These actions are represented as $A_{t-1}$ for past cases and $A_{t}$ for current cases, with rewards granted for different actions.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/0976008e-186f-4bb3-b8cf-152d7d95fc42">
<p>Fig. 2. Action Changes of Agents during the Multi-Agent Reinforcement Learning</p>

## Reward Definition

The reward system in this study, defined as $F1_{delta}$, reflects the performance change before and after each episode. Weaker learners receive higher rewards through additional weights, while stronger learners get less. The final reward is adjusted based on action changes, maintaining a balance between strong and weak learners.

## Data shape

<img width="400" alt="image" src="https://github.com/user-attachments/assets/c51bf333-3622-4d50-8e86-44c1bec1ef08">
<p>Fig. 3. Data Flow Diagram focusing on the Input and Output Data Shape</p>

## Project Directory Structure

The following is the structure of the project directory:


### Files and Their Responsibilities

- **mypackage/**: This folder contains the main modules for data preprocessing, model definition, and reinforcement learning logic.
  - `OO.py`: Contains functions for loading and preprocessing data.
  - `OO.py`: Defines the machine learning models.
  - `OO.py`: Implements reinforcement learning algorithms and Q-learning logic.
  
- **train.py**: This file is used to bring all the components together and run the training process.
