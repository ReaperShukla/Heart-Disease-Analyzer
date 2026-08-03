import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

@st.cache_resource
def load_assets():
    model = joblib.load("knn_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_assets()

st.set_page_config(
    page_title="Heart Disease Risk Analyzer",
    page_icon="❤️",
    layout="centered"
)

st.markdown("""
    <style>
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff7676 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Risk Analyzer")
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 1.1rem;'>Intelligent Diagnostic Screening Interface | Developed by Shashwat</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📋 Patient Demographics")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Biological Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Assessment Type", ["ASY", "ATA", "NAP", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    st.markdown("### 🔬 Clinical & Diagnostic Data")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    resting_ecg = st.selectbox("Resting Electrocardiogram Results", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("Peak Exercise ST Segment Slope", ["Flat", "Up", "Down"])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Generate Diagnostic Risk Assessment"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("---")
    st.markdown("### 📊 Assessment Outcome")
    
    if prediction == 1:
        st.error("🚨 **High Risk of Heart Disease Detected** \n\nThe model identifies vital biomarkers corresponding significantly with clinical heart conditions. Follow-up diagnostic confirmation with a medical professional is recommended.")
    else:
        st.success("🟢 **Low Risk of Heart Disease Detected** \n\nPatient parameters align comfortably with baseline reference populations. Continue regular monitoring and preventative health management.")