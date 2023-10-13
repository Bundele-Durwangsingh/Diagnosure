import  cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('D:/project/Integrated Health Prognosis using Deep Learning/Main/braintumor10Epochs.h5') 
image=cv2.imread('D:\project\Integrated Health Prognosis using Deep Learning\Data\Brain tumor dataset\pred\pred0.jpg')

img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img,axis=0)

result = model.predict(input_img)
print(result)