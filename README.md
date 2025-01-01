# Integrated Healthcare Prognosis System


## Overview
The Integrated Healthcare Prognosis System leverages advanced machine learning (ML) and deep learning (DL) algorithms to predict the likelihood of heart disease, lung cancer, diabetes, and brain tumors. The system is designed for medical staff to input patient data and receive predictive results, which can then be communicated to patients along with preventive measures via SMS.

## Features
- **Heart Disease Prediction**: Utilizes Logistic Regression with an accuracy of 82%.
- **Lung Cancer Prediction**: Utilizes Random Forest with an accuracy of 89%.
- **Diabetes Prediction**: Utilizes Random Forest with an accuracy of 97%.
- **Brain Tumor Detection**: Utilizes Convolutional Neural Network (CNN) with an accuracy of 91%.
- **SMS Notifications**: Sends preventive measures and recommendations to patients via SMS.

## Technology Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Machine Learning Models**: Logistic Regression, Random Forest, Convolutional Neural Network (CNN)
- **Libraries/Frameworks**: TensorFlow, Keras, scikit-learn, OpenCV

## System Architecture

The system architecture involves a user interface where medical staff can input patient data, a backend that processes this data using ML/DL algorithms, and a notification module to send out SMS alerts with preventive measures.

## Implementation Details
### Workflow
1. **User Selection**: The user selects a specific disease for prediction.
2. **Data Input**: The user inputs the patient's details.
3. **Prediction**: The system processes the input data through the corresponding ML/DL algorithm.
4. **Result Display**: The predictive results are displayed to the user.
5. **Notification**: Optionally, the system sends a list of preventive measures to the patient via SMS.

## Output Screens

### Home page
![Home Screen](Image/Home%20screen.png)
![About Us](Image/About%20us.png)
![Expert](Image/Expert%20.png)
![Contact Us](Image/Contact%20us.png)

### Disease Selection
![Disease Selection](Image/Disease%20selection.png)
![Disease Form](Image/Disease%20form.png)
![Prediction Output](Image/Predection%20output.png)
![Brain Tumor Input](Image/Brain%20tumour%20input.png)
![Brain Tumor Output](Image/brain%20tumour%20output.png)

### SMS Notifications
![Send SMS Page](Image/Send%20sms%20page.png)
![Received SMS](Image/Received%20sms.jpg)
![Remedy for Heart Disease](Image/Remedy%20for%20heart%20disease.jpg)
![Remedy for Lung Cancer](Image/Remedy%20for%20lung%20cancer.jpg)
![Remedy for Diabetes](Image/Remedy%20for%20diabetes.jpg)
![Remedy for Brain Tumor](Image/Remendy%20for%20brain%20tumour.jpg)
