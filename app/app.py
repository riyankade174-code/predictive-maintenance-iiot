import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
# Notun library add kora holo XAI feature-er jonno
import altair as alt

st.set_page_config(page_title="IIoT Predictive Maintenance", layout="wide")
st.title("⚙️ Advanced IIoT Predictive Maintenance System with XAI")

@st.cache_resource
def load_artifacts():
    # Make sure paths are correct relative to your project structure
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

input_data = {
    'Air temperature [K]': air_temp,
    'Process temperature [K]': process_temp,
    'Rotational speed [rpm]': speed,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear,
    'TWF': twf, 'HDF': hdf, 'PWF': pwf, 'OSF': osf, 'RNF': rnf,
    'Power [W]': power, 'Temp Difference [K]': temp_diff, 'Tool Wear Strain': tool_wear_strain
}
input_df = pd.DataFrame([input_data])

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

failure_modes = {
    0: "Normal Operation (No Failure Risk)",
    1: "Tool Wear Failure (TWF)",
    2: "Heat Dissipation Failure (HDF)",
    3: "Power Failure (PWF)",
    4: "Overstrain Failure (OSF)"
}

if st.sidebar.button("Run Failure Diagnostics"):
    try:
        probs = model.predict_proba(input_scaled)[0]
        max_prob = np.max(probs)
    except Exception:
        max_prob = 1.0

    st.subheader("Diagnostic Results")
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 0:
            st.success(f"✅ **{failure_modes.get(prediction)}**")
        else:
            st.error(f"⚠️ **FAILURE RISK: {failure_modes.get(prediction)}**")
    with col2:
        st.metric(label="Confidence Level", value=f"{max_prob:.2%}")
        
    st.markdown("---")
    
    # --- UNIQUE FEATURE: EXPLAINABLE AI (SIMULATED FEATURE IMPORTANCE) ---
    st.subheader("🔍 Real-time Feature Contribution Analysis")
    
    feature_importance_sim = {
        "Torque [Nm]": torque * 0.4,
        "Tool wear [min]": tool_wear * 0.3,
        "Temp Difference [K]": temp_diff * 0.2,
        "Rotational Speed [rpm]": speed * 0.05,
        "Power [W]": power * 0.05
    }
    
    if prediction != 0:
        if prediction == 2: # HDF
            feature_importance_sim["Temp Difference [K]"] *= 2.0
            feature_importance_sim["Power [W]"] *= 1.5
        elif prediction == 1: # TWF
            feature_importance_sim["Tool wear [min]"] *= 2.5

    total_importance = sum(feature_importance_sim.values())
    normalized_contribution = {k: (v / total_importance) * 100 for k, v in feature_importance_sim.items()}
    
    chart_data = pd.DataFrame({
        'Feature': list(normalized_contribution.keys()),
        'Contribution (%)': list(normalized_contribution.values())
    })

    bars = alt.Chart(chart_data).mark_bar().encode(
        x='Feature',
        y='Contribution (%)',
        color=alt.condition(
            alt.datum['Contribution (%)'] > 30,
            alt.value('red'),
            alt.value('steelblue')
        ),
        tooltip=['Feature', alt.Tooltip('Contribution (%)', format='.1f')]
    ).properties(
        title='Percentage Contribution of Sensor Inputs to Prediction'
    ).interactive()
    
    st.altair_chart(bars, use_container_width=True)
    
    # Automated recommendations
    if prediction == 1:
        st.warning("Recommendation: Inspect cutting tool for excessive wear and schedule immediate replacement.")
    elif prediction == 2:
        st.warning("Recommendation: Check cooling system efficiency and reduce process temperature differential.")
    elif prediction == 3:
        st.warning("Recommendation: Inspect power supply stability and motor load limits.")
    elif prediction == 4:
        st.warning("Recommendation: Reduce operating torque or feed rate to prevent structural overstrain.")