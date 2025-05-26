import streamlit as st
import pandas as pd
import joblib
import os

def main():
    st.title("Cardiovascular Disease Risk Prediction")

    # Load saved model and scaler
    if not os.path.exists('rf_model.pkl') or not os.path.exists('scaler.pkl'):
        st.error("Model or scaler not found. Please run the training script.")
        return

    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')

    st.sidebar.header("Enter Patient Data")

    # Include ID input
    patient_id = st.sidebar.number_input("Patient ID", min_value=0, value=1)

    # User inputs for model features
    age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=44)
    gender = st.sidebar.selectbox("Gender (1=male, 2=female)", options=[1, 2], index=0)
    height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=153)
    weight = st.sidebar.number_input("Weight (kg)", min_value=20, max_value=300, value=93)
    ap_hi = st.sidebar.number_input("Systolic BP (ap_hi)", min_value=50, max_value=250, value=140)
    ap_lo = st.sidebar.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=150, value=90)
    cholesterol = st.sidebar.selectbox("Cholesterol (1=normal, 2=above normal, 3=well above normal)", options=[1, 2, 3], index=0)
    gluc = st.sidebar.selectbox("Glucose (1=normal, 2=above normal, 3=well above normal)", options=[1, 2, 3], index=0)
    smoke = st.sidebar.selectbox("Smoke (0=No, 1=Yes)", options=[0, 1], index=0)
    alco = st.sidebar.selectbox("Alcohol intake (0=No, 1=Yes)", options=[0, 1], index=0)
    active = st.sidebar.selectbox("Physical Activity (0=No, 1=Yes)", options=[0, 1], index=1)

    # Create input dict with one-hot encoded variables
    input_dict = {
        'id': patient_id,
        'age': age,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'gender_2': 1 if gender == 2 else 0,
        'cholesterol_2': 1 if cholesterol == 2 else 0,
        'cholesterol_3': 1 if cholesterol == 3 else 0,
        'gluc_2': 1 if gluc == 2 else 0,
        'gluc_3': 1 if gluc == 3 else 0,
        'smoke_1': 1 if smoke == 1 else 0,
        'alco_1': 1 if alco == 1 else 0,
        'active_1': 1 if active == 1 else 0
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure feature columns match the model's expected order
    feature_cols = model.feature_names_in_
    try:
        input_df = input_df[feature_cols]
    except KeyError as e:
        st.error(f"Input data is missing expected columns: {e}")
        return

    # Apply the same scaler as during training
    input_scaled = scaler.transform(input_df)

    if st.sidebar.button("Predict Risk"):
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")
        st.write(f"Patient ID: {patient_id}")
        st.write(f"Prediction: {'At Risk' if prediction == 1 else 'Not at Risk'}")
        st.write(f"Risk Probability: {prob:.2%}")

if __name__ == "__main__":
    main()
