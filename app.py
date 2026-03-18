import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Set Page Config
st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁", layout="wide")

# Load Artifacts
@st.cache_resource
def load_artifacts():
    base_path = r"C:\PYTHON PROGRAMMING\Lungs Cancer Prediction"
    model = joblib.load(os.path.join(base_path, "lung_cancer_rf_model.joblib"))
    scaler = joblib.load(os.path.join(base_path, "lung_cancer_scaler.joblib"))
    encoder = joblib.load(os.path.join(base_path, "lung_cancer_encoder.joblib"))
    return model, scaler, encoder

try:
    model, scaler, encoder = load_artifacts()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and Description
st.title("🫁 Lung Cancer Prediction AI")
st.markdown("""
This application uses a Machine Learning model to predict the risk level of lung cancer based on various health factors and habits.
**Please adjust the sliders in the sidebar to match the patient's data.**
""")

# Sidebar Inputs
st.sidebar.header("Patient Data")

def user_input_features():
    age = st.sidebar.slider("Age", 1, 100, 30)
    gender = st.sidebar.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    
    st.sidebar.markdown("### Environmental & Habits (Scale 1-9)")
    air_pollution = st.sidebar.slider("Air Pollution", 1, 9, 3)
    alcohol_use = st.sidebar.slider("Alcohol Use", 1, 9, 3)
    dust_allergy = st.sidebar.slider("Dust Allergy", 1, 9, 3)
    occupational_hazards = st.sidebar.slider("Occupational Hazards", 1, 9, 3)
    genetic_risk = st.sidebar.slider("Genetic Risk", 1, 9, 3)
    chronic_lung_disease = st.sidebar.slider("Chronic Lung Disease", 1, 9, 3)
    balanced_diet = st.sidebar.slider("Balanced Diet", 1, 9, 3)
    obesity = st.sidebar.slider("Obesity", 1, 9, 3)
    smoking = st.sidebar.slider("Smoking", 1, 9, 3)
    passive_smoker = st.sidebar.slider("Passive Smoker", 1, 9, 3)
    
    st.sidebar.markdown("### Symptoms (Scale 1-9)")
    chest_pain = st.sidebar.slider("Chest Pain", 1, 9, 3)
    coughing_of_blood = st.sidebar.slider("Coughing of Blood", 1, 9, 3)
    fatigue = st.sidebar.slider("Fatigue", 1, 9, 3)
    weight_loss = st.sidebar.slider("Weight Loss", 1, 9, 3)
    shortness_of_breath = st.sidebar.slider("Shortness of Breath", 1, 9, 3)
    wheezing = st.sidebar.slider("Wheezing", 1, 9, 3)
    swallowing_difficulty = st.sidebar.slider("Swallowing Difficulty", 1, 9, 3)
    clubbing_of_finger_nails = st.sidebar.slider("Clubbing of Finger Nails", 1, 9, 3)
    frequent_cold = st.sidebar.slider("Frequent Cold", 1, 9, 3)
    dry_cough = st.sidebar.slider("Dry Cough", 1, 9, 3)
    snoring = st.sidebar.slider("Snoring", 1, 9, 3)
    
    data = {
        'Age': age,
        'Gender': gender,
        'Air Pollution': air_pollution,
        'Alcohol use': alcohol_use,
        'Dust Allergy': dust_allergy,
        'OccuPational Hazards': occupational_hazards,
        'Genetic Risk': genetic_risk,
        'chronic Lung Disease': chronic_lung_disease,
        'Balanced Diet': balanced_diet,
        'Obesity': obesity,
        'Smoking': smoking,
        'Passive Smoker': passive_smoker,
        'Chest Pain': chest_pain,
        'Coughing of Blood': coughing_of_blood,
        'Fatigue': fatigue,
        'Weight Loss': weight_loss,
        'Shortness of Breath': shortness_of_breath,
        'Wheezing': wheezing,
        'Swallowing Difficulty': swallowing_difficulty,
        'Clubbing of Finger Nails': clubbing_of_finger_nails,
        'Frequent Cold': frequent_cold,
        'Dry Cough': dry_cough,
        'Snoring': snoring
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main Panel Display
st.subheader("Patient Profile")
st.write(input_df)

if st.button("Predict Risk Level"):
    # Preprocess
    scaled_features = scaler.transform(input_df)
    
    # Predict
    prediction_encoded = model.predict(scaled_features)
    prediction = encoder.inverse_transform(prediction_encoded)
    probability = model.predict_proba(scaled_features)
    
    st.subheader("Prediction Result")
    result = prediction[0]
    
    if result == "Low":
        st.success(f"Risk Level: **{result}**")
        st.balloons()
    elif result == "Medium":
        st.warning(f"Risk Level: **{result}**")
    else:
        st.error(f"Risk Level: **{result}**")
        
    st.write("Confidence Scores:")
    probs_df = pd.DataFrame(probability, columns=encoder.classes_)
    st.bar_chart(probs_df.T)

st.markdown("---")
st.markdown("*Disclaimer: This tool is for educational purposes only and should not replace professional medical advice.*")
