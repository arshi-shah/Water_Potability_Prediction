import streamlit as st
import pandas as pd
from joblib import load

# Load your trained model (trained on all 9 features)
model = load('./models/water_model_all_features.joblib')



st.title("Water Potability Prediction App")
st.header("Enter Water Quality Parameters")

# Input fields for all 9 features
import streamlit as st


# Sliders for all 9 features, no units in labels
ph = st.slider("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.slider("Hardness", min_value=0.0, max_value=500.0, value=150.0, step=0.1)
solids = st.slider("Solids", min_value=0.0, max_value=100000.0, value=20000.0, step=0.1)
chloramines = st.slider("Chloramines", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
conductivity = st.slider("Conductivity", min_value=0.0, max_value=800.0, value=300.0, step=0.1)
organic_carbon = st.slider("Organic Carbon", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
sulfate = st.slider("Sulfate", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
trihalomethanes = st.slider("Trihalomethanes", min_value=0.0, max_value=150.0, value=50.0, step=0.1)
turbidity = st.slider("Turbidity", min_value=0.0, max_value=10.0, value=2.0, step=0.1)


# Predict button
if st.button("Predict Water Potability"):
    # Prepare input DataFrame in the same order as training
    input_df = pd.DataFrame([{
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Sulfate': sulfate,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Show result
    st.header("Prediction Result")
    if prediction == 1:
        st.error("Warning: Water may NOT be safe to drink.")
    else:
        st.success("Water is likely safe to drink.")
