import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("logistic_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease")

# Input fields (modify based on your dataset)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=400)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, thalach, oldpeak]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")
