import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def train_and_save_model():
    print("1. Loading dataset...")
    # Path to your dataset (update filename if needed)
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'ai4i2020.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please place your CSV file in a 'data' folder.")
        return

    df = pd.read_csv(data_path)

    print("2. Feature Engineering...")
    # Creating custom interaction features used in the dashboard
    df['Power [W]'] = df['Rotational speed [rpm]'] * df['Torque [Nm]'] * (2 * np.pi / 60)
    df['Temp Difference [K]'] = df['Process temperature [K요일]' if 'Process temperature [K요일]' in df.columns else 'Process temperature [K]'] - df['Air temperature [K]']
    df['Tool Wear Strain'] = df['Tool wear [min]'] * df['Torque [Nm]']

    # Defining target mapping (Multi-class failure diagnosis based on failure flags)
    # 0: Normal, 1: TWF, 2: HDF, 3: PWF, 4: OSF
    def get_failure_mode(row):
        if row['TWF'] == 1: return 1
        elif row['HDF'] == 1: return 2
        elif row['PWF'] == 1: return 3
        elif row['OSF'] == 1: return 4
        else: return 0

    df['Failure_Mode'] = df.apply(get_failure_mode, axis=1)

    # Selecting features for training
    feature_cols = [
        'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
        'Power [W]', 'Temp Difference [K]', 'Tool Wear Strain'
    ]
    
    X = df[feature_cols]
    y = df['Failure_Mode']

    print("3. Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("5. Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print("6. Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)

    print("7. Evaluating Model...")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    print("8. Saving model and scaler artifacts...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Training complete! Model and scaler saved successfully inside the 'models/' folder.")

if __name__ == "__main__":
    train_and_save_model()