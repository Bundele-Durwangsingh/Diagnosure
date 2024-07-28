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

## Results
- **Heart Disease Prediction Accuracy**: 82%
- **Lung Cancer Prediction Accuracy**: 89%
- **Diabetes Prediction Accuracy**: 97%
- **Brain Tumor Detection Accuracy**: 91%

## Future Work
- **Expansion**: Integrate more diseases for prediction.
- **Mobile App Development**: Develop a mobile application for easier access.
- **Integration with EHR**: Connect with electronic health records for seamless data flow.
- **Algorithm Improvement**: Enhance existing algorithms and develop new ones using deep learning.
- **Personalized Treatment**: Provide personalized treatment recommendations based on predictive results.

## Conclusion
The Integrated Healthcare Prognosis System aims to assist in the early detection of critical diseases, thereby reducing the burden on healthcare systems and promoting patient well-being through timely and preventive measures.
