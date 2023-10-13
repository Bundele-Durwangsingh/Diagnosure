import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("D:\project\major project implementation\Data\survey lung cancer.csv")
df.head()

ds= pd.DataFrame(df)
ds=ds.drop(['PEER_PRESSURE'], axis=1)
ds.shape

ds['GENDER']= ds['GENDER'].apply({'M':1, 'F':0}.get)
ds.head(10)
ds['LUNG_CANCER']= ds['LUNG_CANCER'].apply({'YES':1, 'NO':0}.get)
ds.head()

X = ds.drop('LUNG_CANCER', axis=1)
y = ds['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

RF=RandomForestClassifier()
RF.fit(X_train, y_train)
pred_rf = RF.predict(X_test)
print("Accuracy score for Random Forest")
print(RF.score(X_test,y_test))
input_data = (0,59,1,1,1,1,2,1,2,1,2,2,1,2)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = RF.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have a lung cancer')
else:
    print('The Person might have lung cancer')