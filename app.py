import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# ----------------------------------
# Load Model and Metadata
# ----------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("laptop_price_model.pkl")
    scaler = joblib.load("scaler.pkl")

    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    with open("dropdowns.pkl", "rb") as f:
        dropdowns = pickle.load(f)

    return model, scaler, model_columns, dropdowns


model, scaler, model_columns, dropdowns = load_files()

# ----------------------------------
# UI
# ----------------------------------

st.title("💻 Laptop Price Prediction")
st.write("Enter laptop specifications to estimate its price.")

st.subheader("🔧 Laptop Specifications")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", dropdowns["Company"])
    type_name = st.selectbox("Laptop Type", dropdowns["TypeName"])
    cpu = st.selectbox("CPU Brand", dropdowns["Cpu_brand"])
    gpu = st.selectbox("GPU Brand", dropdowns["Gpu_brand"])
    os = st.selectbox("Operating System", dropdowns["OpSys"])

with col2:
    ram = st.selectbox("RAM (GB)", dropdowns["Ram"])
    inches = st.number_input("Screen Size (Inches)", 10.0, 20.0, step=0.1)
    ssd = st.number_input("SSD (GB)", 0, 2000, step=128)
    hdd = st.number_input("HDD (GB)", 0, 2000, step=256)
    weight = st.number_input("Weight (kg)", 0.5, 5.0, step=0.1)

# ----------------------------------
# Prediction
# ----------------------------------

if st.button("🔮 Predict Price"):

    input_data = {
        "Company": company,
        "TypeName": type_name,
        "Cpu_brand": cpu,
        "Gpu_brand": gpu,
        "OpSys": os,
        "Ram": ram,
        "Inches": inches,
        "SSD": ssd,
        "HDD": hdd,
        "Weight": weight
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encoding
    encoded_df = pd.get_dummies(input_df)

    # Align columns
    encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

    try:
        # Scale features
        scaled_input = scaler.transform(encoded_df)

        # Predict
        prediction = model.predict(scaled_input)
        price = prediction[0]

        st.success(f"💰 Estimated Laptop Price: ₹{int(price):,}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning")