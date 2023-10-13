import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

RF=RandomForestClassifier()
RF.fit(X_train, y_train)
pred_rf = RF.predict(X_test)
print("Accuracy score for Random Forest")
print(RF.score(X_test,y_test))
input_data = (0,44,0,0,1,19.31,6.5,200)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = RF.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have a diabetes')
else:
    print('The Person might have diabetes')