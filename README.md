# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score . 


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: D.HARITHA.
RegisterNumber: 212225040118
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path =("C:/Users/acer/Downloads/Employee.csv")
df = pd.read_csv(file_path)
print("Dataset Columns:", df.columns.tolist())

if 'salary' in df.columns:
    salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
    df['salary'] = df['salary'].map(salary_mapping)

df = pd.get_dummies(df, drop_first=True)

target_col = 'left' 
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=['Stayed', 'Left'], 
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Employee Churn Decision Tree Logic")
plt.show()
```

## Output:

<img width="813" height="636" alt="Screenshot 2026-05-13 091542" src="https://github.com/user-attachments/assets/77f2fecb-f445-4d8b-b4ba-03540a969513" />

<img width="1387" height="623" alt="Screenshot 2026-05-13 091557" src="https://github.com/user-attachments/assets/94a1811d-e4f4-4316-af77-22233ccbfefd" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
