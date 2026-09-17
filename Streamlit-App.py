import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model('salary_model.keras')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

st.title("Salary Prediction App")
st.write("Enter customer details to predict estimated salary")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
exited = st.selectbox("Exited", [0, 1])

if st.button("Predict Salary"):
    input_df = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'Exited': exited
    }])

    input_encoded = encoder.transform(input_df)
    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)
    st.success(f"Predicted Estimated Salary: ${prediction[0][0]:,.2f}")