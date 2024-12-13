import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Streamlit app
st.title("Heart Disease Prediction Tool")

# Add some introductory text with better formatting
st.markdown("""
This app predicts the likelihood of heart disease based on the details you provide.
Fill in the following details, and we will predict the risk for you.
""")

# Create two columns for input fields on one line
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

with col2:
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])

# Create two more columns for the remaining input fields
col3, col4 = st.columns(2)

with col3:
    cp = st.selectbox("Chest Pain Type (0: Typical, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic)", [0, 1, 2, 3])

with col4:
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)

col5, col6 = st.columns(2)

with col5:
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)

with col6:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (1: True, 0: False)", [0, 1])

col7, col8 = st.columns(2)

with col5:
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])

with col6:
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)


col9, col10 = st.columns(2)

with col9:
    exang = st.selectbox("Exercise-Induced Angina (1: Yes, 0: No)", [0, 1])

with col10:
    oldpeak = st.number_input("ST Depression Induced by Exercise (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)


col11, col12 = st.columns(2)

with col11:
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])

with col12:
    ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3])


thal = st.selectbox("Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)", [1, 2, 3])

# Prediction
if st.button("Predict"):
    # Prepare the input array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Make the prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of class 1 (Heart Disease)

    # Display the result
    st.subheader("Prediction Results:")
    if prediction == 1:
        st.write(f"**High Risk of Heart Disease**")
    else:
        st.write(f"**Low Risk of Heart Disease**")
    
    st.write(f"Confidence Level: {probability:.2f}")
