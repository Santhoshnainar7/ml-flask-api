import streamlit as st
import joblib

# Load trained ML model
model = joblib.load("house_model.pkl")

# Page title
st.title("House Price Prediction")
st.write("Enter house details to predict the price")

# User inputs
area = st.number_input("Area (sq.ft)", min_value=100)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, step=1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Predicted House Price: {prediction[0]:.2f}")
