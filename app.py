import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
model = joblib.load("startup_model.pkl")
le_industry = joblib.load("le_industry.pkl")
le_city = joblib.load("le_city.pkl")

st.title("🚀 Startup Success Predictor")

# User inputs
industry = st.selectbox("Select Industry Vertical:", le_industry.classes_)
city = st.selectbox("Select City Location:", le_city.classes_)

if st.button("Predict Success"):
    # Encode inputs
    industry_encoded = le_industry.transform([industry])
    city_encoded = le_city.transform([city])
    
    # Prepare dataframe for prediction
    X_new = pd.DataFrame({
        'Industry Vertical': industry_encoded,
        'City  Location': city_encoded
    })
    
    # Predict probability of success
    success_prob = model.predict_proba(X_new)[:,1][0]
    
    st.success(f"Predicted Probability of Success: {success_prob*100:.2f}%")
