import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import pytz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Page Configuration
st.set_page_config(page_title="Advanced AI4I Predictive Maintenance", layout="wide")

# --- SIDEBAR: Date, Time & Location Configuration ---
st.sidebar.header("⚙️ System Environment")

# Current Date & Time (IST / Local)
try:
    IST = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(IST).strftime("%Y-%m-%d | %H:%M:%S")
except:
    current_time = datetime.now().strftime("%Y-%m-%d | %H:%M:%S")

st.sidebar.markdown(f"**📅 Date & Time:** `{current_time}`")

# Plant Location Selector
plant_location = st.sidebar.selectbox(
    "🏭 Select Facility / Location",
    ["Plant A - Unit 1 (Kolkata)", "Plant B - Unit 2 (Durgapur)", "Plant C - Assembly Line 3"]
)

st.title("⚙️ AI4I 2020 Advanced Predictive Maintenance Dashboard")
st.write(f"Monitoring equipment health and diagnostics for **{plant_location}** using an optimized Random Forest classifier.")

# Function to train and save model if not already present
@st.cache_resource
def load_or_train_model():
    model_path = 'models/random_forest_model.pkl'
    scaler_path = 'models/scaler.pkl'
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'ai4i2020.csv')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.info("Training model for the first time. Please wait...")
        df = pd.read_csv(data_path)
        
        # Feature Engineering
        df['Power [W]'] = df['Rotational speed [rpm]'] * df['Torque [Nm]'] * (2 * np.pi / 60)
        temp_col = 'Process temperature [K]' if 'Process temperature [K]' in df.columns else df.columns[df.columns.str.contains('Process temperature')][0]
        df['Temp Difference [K]'] = df[temp_col] - df['Air temperature [K]']
        df['Tool Wear Strain'] = df['Tool wear [min]'] * df['Torque [Nm]']

        def get_failure_mode(row):
            if row['TWF'] == 1: return 1
            elif row['HDF'] == 1: return 2
            elif row['PWF'] == 1: return 3
            elif row['OSF'] == 1: return 4
            else: return 0

        df['Failure_Mode'] = df.apply(get_failure_mode, axis=1)

        feature_cols = [
            'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
            'Power [W]', 'Temp Difference [K]', 'Tool Wear Strain'
        ]
        
        X = df[feature_cols]
        y = df['Failure_Mode']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(X_train_resampled, y_train_resampled)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load model & data
model, scaler = load_or_train_model()

# Session State for storing real-time prediction logs
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

data_path = os.path.join(os.path.dirname(__file__), 'data', 'ai4i2020.csv')
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Dashboard Navigation")
    menu = st.sidebar.selectbox("Choose View", ["Real-time Prediction", "Dataset Overview", "Feature Importance Analysis", "Prediction Logs"])

    if menu == "Real-time Prediction":
        st.subheader("🔧 Machine Failure Prediction Panel")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            air_temp = st.number_input("Air temperature [K]", value=298.0)
            process_temp = st.number_input("Process temperature [K]", value=308.0)
            rot_speed = st.number_input("Rotational speed [rpm]", value=1550)
        with col2:
            torque = st.number_input("Torque [Nm]", value=40.0)
            tool_wear = st.number_input("Tool wear [min]", value=0)
            twf = st.selectbox("TWF (Tool Wear Failure)", [0, 1])
        with col3:
            hdf = st.selectbox("HDF (Heat Dissipation Failure)", [0, 1])
            pwf = st.selectbox("PWF (Power Failure)", [0, 1])
            osf = st.selectbox("OSF (Overstrain Failure)", [0, 1])
            rnf = st.selectbox("RNF (Random Failure)", [0, 1])

        if st.button("Run Diagnostic Prediction"):
            power = rot_speed * torque * (2 * np.pi / 60)
            temp_diff = process_temp - air_temp
            tool_strain = tool_wear * torque

            input_data = pd.DataFrame([[
                air_temp, process_temp, rot_speed, torque, tool_wear,
                twf, hdf, pwf, osf, rnf, power, temp_diff, tool_strain
            ]], columns=[
                'Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
                'Power [W]', 'Temp Difference [K]', 'Tool Wear Strain'
            ])

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            failure_types = {
                0: "Normal (No Failure)", 
                1: "Tool Wear Failure (TWF)", 
                2: "Heat Dissipation Failure (HDF)", 
                3: "Power Failure (PWF)", 
                4: "Overstrain Failure (OSF)"
            }
            
            result_text = failure_types[prediction]
            confidence = float(np.max(prediction_proba) * 100)

            # Log the prediction with timestamp and location
            log_entry = {
                "Timestamp": current_time,
                "Location": plant_location,
                "Prediction": result_text,
                "Confidence (%)": f"{confidence:.2f}%"
            }
            st.session_state.prediction_history.insert(0, log_entry)

            if prediction == 0:
                st.success(f"Status: **{result_text}** (Confidence: {confidence:.2f}%)")
            else:
                st.error(f"Alert! Detected: **{result_text}** (Confidence: {confidence:.2f}%)")

    elif menu == "Dataset Overview":
        st.subheader("📊 Dataset Preview & Statistics")
        st.dataframe(df.head(10))
        st.metric("Total Plant Records Loaded", df.shape[0])

    elif menu == "Feature Importance Analysis":
        st.subheader("📈 Model Feature Importance (Random Forest)")
        feature_cols = [
            'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
            'Power [W]', 'Temp Difference [K]', 'Tool Wear Strain'
        ]
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        st.bar_chart(imp_df.set_index('Feature'))

    elif menu == "Prediction Logs":
        st.subheader("📋 Recent Diagnostic Logs (Session History)")
        if len(st.session_state.prediction_history) > 0:
            log_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(log_df)
        else:
            st.info("No predictions made yet in this session. Run a prediction from the Real-time panel!")
else:
    st.error("Dataset 'ai4i2020.csv' not found inside 'data/' folder!")