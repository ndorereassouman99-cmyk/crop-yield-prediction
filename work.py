# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 23:17:49 2025

@author: user
"""
import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Step 1: Load saved model and encoders
# -----------------------------
with open("25RP18809 by Random Forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("le_field.pkl", "rb") as f:
    le_field = pickle.load(f)

with open("le_crop.pkl", "rb") as f:
    le_crop = pickle.load(f)

# -----------------------------
# Step 2: Streamlit UI
# -----------------------------
st.title("Crop Yield Prediction")

st.write("Enter field and crop information:")

field_id_input = st.text_input("Field ID (e.g., Field_23)")
crop_type_input = st.text_input("Crop Type (e.g., Wheat)")
NDVI = st.number_input("NDVI value", value=0.5)
GNDVI = st.number_input("GNDVI value", value=0.5)
NDWI = st.number_input("NDWI value", value=0.5)
SAVI = st.number_input("SAVI value", value=0.5)
soil_moisture = st.number_input("Soil Moisture", value=0.1)
temperature = st.number_input("Temperature", value=25.0)
rainfall = st.number_input("Rainfall", value=5.0)

# -----------------------------
# Step 3: Encode categorical features
# -----------------------------
if st.button("Predict Yield"):
    try:
        field_encoded = le_field.transform([field_id_input])[0]
    except ValueError:
        st.warning(f"'{field_id_input}' not seen during training. Using 0 as default.")
        field_encoded = 0

    try:
        crop_encoded = le_crop.transform([crop_type_input])[0]
    except ValueError:
        st.warning(f"'{crop_type_input}' not seen during training. Using 0 as default.")
        crop_encoded = 0

    # -----------------------------
    # Step 4: Prepare input DataFrame
    # -----------------------------
    input_data = pd.DataFrame([[
        field_encoded, crop_encoded, NDVI, GNDVI, NDWI,
        SAVI, soil_moisture, temperature, rainfall
    ]], columns=[
        'field_id_encoded', 'crop_type_encoded', 'NDVI', 'GNDVI', 'NDWI',
        'SAVI', 'soil_moisture', 'temperature', 'rainfall'
    ])

    # -----------------------------
    # Step 5: Predict
    # -----------------------------
    predicted_yield = model.predict(input_data)[0]
    st.success(f"Predicted Crop Yield: {predicted_yield:.2f} units")