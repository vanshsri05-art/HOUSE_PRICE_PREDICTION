import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

# Inputs
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Price: ₹ {int(prediction[0])}")