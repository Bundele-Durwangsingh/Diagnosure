from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

model_hdp = pickle.load(open('model.pkl', 'rb'))
model_lcp = pickle.load(open('lungs_model.pkl', 'rb'))
model_dp = pickle.load(open('diabetes_model.pkl', 'rb'))
model =load_model('braintumor10Epochs.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart.html')
def heart():
    return render_template('heart.html')


@app.route('/hdp_predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cp = int(request.form.get("cp"))
    trestbps = int(request.form.get("trestbps"))
    chol = int(request.form.get("chol"))
    fbs = int(request.form.get("fbs"))
    restecg = int(request.form.get("restecg"))
    thalach = int(request.form.get("thalach"))
    exang = int(request.form.get("exang"))
    oldpeak = int(request.form.get("oldpeak"))
    slope = int(request.form.get("slope"))
    ca = int(request.form.get("ca"))
    thal = int(request.form.get("thal"))
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model_hdp.predict(input_data_reshaped)
    print(prediction)
    if prediction == 1:
        return render_template('heart.html', label=1)
    else:
        return render_template('heart.html', label=-1)

@app.route('/lungs.html')
def lcp():
    return render_template('lungs.html')


@app.route('/lcp_predict', methods=['POST'])
def lcp_predict():
    gen = int(request.form.get('gen'))
    age = int(request.form.get('age'))
    smoke = int(request.form.get("smoke"))
    ylw_fin = int(request.form.get("ylw_fin"))
    anx = int(request.form.get("anx"))
    cd = int(request.form.get("cd"))
    fati = int(request.form.get("fati"))
    alg = int(request.form.get("alg"))
    whz = int(request.form.get("whz"))
    alc = int(request.form.get("alc"))
    cough = int(request.form.get("cough"))
    sb = int(request.form.get("sb"))
    sd = int(request.form.get("sd"))
    cp = int(request.form.get("cp"))
    input_data = (gen,age,smoke,ylw_fin,anx,cd,fati,alg,whz,alc,cough,sb,sd,cp)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model_lcp.predict(input_data_reshaped)
    print(prediction)
    if prediction == 1:
        return render_template('lungs.html', label=1)
    else:
        return render_template('lungs.html', label=-1)


@app.route('/diabetes.html')
def dp():
    return render_template('diabetes.html')


@app.route('/dp_predict', methods=['POST'])
def dp_predict():
    gen = int(request.form.get('gen'))
    age = int(request.form.get('age'))
    hyper = int(request.form.get("hyper"))
    hd = int(request.form.get("hd"))
    sh = int(request.form.get("sh"))
    bmi = float(request.form.get("bmi"))
    hl = float(request.form.get("hl"))
    bgl = int(request.form.get("bgl"))

    input_data = (gen,age,hyper,hd,sh,bmi,hl,bgl)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model_dp.predict(input_data_reshaped)
    print(prediction)
    if prediction == 1:
        return render_template('diabetes.html', label=1)
    else:
        return render_template('diabetes.html', label=-1)

print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
 # Get the index of the class with the highest probability
    return result



@app.route('/braintumor.html',methods=['GET'])
def index():
    return render_template('braintumor.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
