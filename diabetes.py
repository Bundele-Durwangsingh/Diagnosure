import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("D:\project\Integrated Health Prognosis using Deep Learning\Data\diabetes_prediction_dataset.csv")
df['gender']= df['gender'].apply({'Male':1, 'Female':0}.get)
df['smoking_history']= df['smoking_history'].apply({'No Info':0,'never':1,'former':2,'not current':3,'ever':4,'current':5}.get)
df['age'] = df['age'].astype(np.int64)
df['gender'] = df['gender'].fillna(0).astype(np.int64)

X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

DP_RF=RandomForestClassifier()
DP_RF.fit(X_train, y_train)
pred_rf = DP_RF.predict(X_test)
pickle.dump(DP_RF,open('diabetes_model.pkl','wb'))
