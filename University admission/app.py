import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Admission_prediction.joblib')

st.title("🎓 Admission Chance Predictor")

st.markdown("Enter the following academic details:")

# Inputs that match model training features
gre_score = st.slider("GRE Score", 260, 340, 300)
toefl_score = st.slider("TOEFL Score", 0, 120, 100)
university_rating = st.slider("University Rating", 1, 5, 3)
sop = st.slider("SOP Strength", 1.0, 5.0, 3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)
cgpa = st.slider("CGPA", 6.0, 10.0, 8.5)

if st.button("Predict Admission Chance"):
    # Only 6 features
    features = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa]])
    prediction = model.predict(features)[0]
    st.success(f"🎯 Predicted Chance of Admission: {prediction:.2f}")
