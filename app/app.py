import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("⚙️ IIoT Predictive Maintenance System")

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), '../models/random_forest_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_artifacts()

st.sidebar.header("Industrial Sensor Inputs")
air_temp = st.sidebar.number_input("Air Temperature [K]", 290.0, 310.0, 300.0)
process_temp = st.sidebar.number_input("Process Temperature [K]", 300.0, 320.0, 310.0)
speed = st.sidebar.slider("Rotational Speed [rpm]", 1000, 3000, 1500)
torque = st.sidebar.slider("Torque [Nm]", 10.0, 90.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, 100)

twf = hdf = pwf = osf = rnf = 0
power = speed * torque * (2 * np.pi / 60)
temp_diff = process_temp - air_temp
tool_wear_strain = tool_wear * torque

if st.sidebar.button("Run Failure Diagnostics"):
    input_df = pd.DataFrame([[
        air_temp, process_temp, speed, torque, tool_wear,
        twf, hdf, pwf, osf, rnf,
        power, temp_diff, tool_wear_strain
    ]], columns=[
        'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
        'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
        'Power [W]', 'Temp Difference [K]', 'Tool Wear Strain'
    ])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Diagnostic Results")
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("⚠️ **HIGH FAILURE RISK DETECTED**")
        else:
            st.success("✅ **EQUIPMENT OPERATING NORMALLY**")
    with col2:
        st.metric(label="Failure Probability", value=f"{prob:.2%}")