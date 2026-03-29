import streamlit as st
import requests

st.set_page_config(page_title="Health Risk Predictor", layout="centered")

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
    <style>
    .main-title {
        font-size: 200px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
    }
    .card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title('🩺 Health Risk Predictor')

st.write("Enter your health details:")

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

with col2:
    glucose = st.number_input("Glucose Level", 50.0, 250.0, 100.0)
    bp = st.number_input("Blood Pressure", 70.0, 200.0, 120.0)

smoking = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"])

# =========================
# BUTTON
# =========================
if st.button(" Predict Risk"):

    data = {
        "age": age,
        "bmi": bmi,
        "glucose_level": glucose,
        "blood_pressure": bp,
        "smoking_status": smoking
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)

        if response.status_code == 200:
            result = response.json()

            risk = result["risk"]
            confidence = result["confidence"]
            reasons = result["reasons"]

            # =========================
            # RESULT CARD
            # =========================
            st.markdown('<div class="card">', unsafe_allow_html=True)

            if risk == "High":
                st.error(" HIGH RISK")
            else:
                st.success(" LOW RISK")

            st.write(f"**Confidence:** {confidence}")

            # Progress bar
            st.progress(confidence)

            st.write("###  Reasons:")
            for r in reasons:
                st.write(f"- {r}")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}")