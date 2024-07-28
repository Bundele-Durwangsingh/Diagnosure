import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


heart_data = pd.read_csv("heart_disease_data.csv")
X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy on Training data : ", training_data_accuracy)
print("Accuracy on Test data : ", test_data_accuracy)

pickle.dump(model, open("heart_model.pkl", "wb"))
