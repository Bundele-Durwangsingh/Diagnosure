import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical

image_dir = r"D:\project\Integrated Health Prognosis using Deep Learning\Data\Brain tumor dataset"

no_tumor = os.listdir(os.path.join(image_dir, 'no'))
yes_tumor = os.listdir(os.path.join(image_dir, 'yes'))
dataset = []
label = []
input_size=64

for image_name in no_tumor:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(os.path.join(image_dir, 'no', image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size,input_size))
        dataset.append(np.array(image))
        label.append(0)

for image_name in yes_tumor:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(os.path.join(image_dir, 'yes', image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size,input_size))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)

X_train, X_test, y_train, y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)

X_test=normalize(X_test,axis=1)
X_train=normalize(X_train,axis=1)
y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(input_size,input_size,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=16,verbose=1,epochs=10,validation_data=(X_test,y_test),shuffle=False)
model.save('braintumor10Epochscategorical.h5')