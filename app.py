import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("startup_model.pkl")

# Load LabelEncoders (if needed, we save them too)
le_industry = joblib.load("le_industry.pkl")
le_city = joblib.load("le_city.pkl")

st.title("Startup Success Predictor 🚀")

# User inputs
industry = st.selectbox("Industry Vertical", le_industry.classes_)
city = st.selectbox("City  Location", le_city.classes_)

# Predict button
if st.button("Predict Success"):
    # Encode inputs
    industry_encoded = le_industry.transform([industry])
    city_encoded = le_city.transform([city])
    
    # Prepare input dataframe
    new_startup = pd.DataFrame({
        'Industry Vertical': industry_encoded,
        'City  Location': city_encoded
    })
    
    # Predict probability
    success_prob = model.predict_proba(new_startup)[:,1][0]
    
    st.success(f"Predicted probability of success: {success_prob*100:.2f}%")
