# =============================================
# Smart Diabetes Prediction System (Streamlit)
# =============================================
import streamlit as st
import joblib
import numpy as np

# ---------------------------------------------
# Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="P_CAT_AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------
# Custom CSS for a cleaner, smarter UI
# ---------------------------------------------
st.markdown(
    """
    <style>
        .main {background-color: #f9fbfd;}
        .title-text {text-align:center; color:#2C3E50; font-weight:700;}
        .subtitle-text {text-align:center; color:#5D6D7E; font-size:16px;}
        .card {
            background-color:white;
            padding:20px;
            border-radius:12px;
            box-shadow:0px 4px 12px rgba(0,0,0,0.05);
            margin-bottom:20px;
        }
        .footer {text-align:center; color:#95A5A6; font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------
# Load Model & Scaler
# ---------------------------------------------
@st.cache_resource

def load_artifacts():
    model = joblib.load("25RP18236_model.joblib")
    scaler = joblib.load("25RP18236_scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------------------
# Sidebar (Professional & Minimal)
# ---------------------------------------------
with st.sidebar:
    st.markdown("## 📌 CAT")
    st.info(
        """
        **System:** Diabetes Prediction   
        **REG_NUMBER:** 25RP18236 
        **MECHATRONICS**
        """
    )
    

# ---------------------------------------------
# Input Section
# ---------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🧾 Patient Health Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, help="Number of times pregnant")
        glucose = st.number_input("🩸 Glucose Level ", min_value=0, help="Plasma glucose concentration")
        blood_pressure = st.number_input("💓 Blood Pressure ", min_value=0)
        skin_thickness = st.number_input("📏 Skin Thickness ", min_value=0)

    with col2:
        insulin = st.number_input("💉 Insulin Level ", min_value=0)
        bmi = st.number_input("⚖️ BMI ", min_value=0.0, format="%.2f")
        dpf = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0, format="%.3f")
        age = st.number_input("🎂 Age ", min_value=1, max_value=120)

    submit = st.form_submit_button("🔍 Predict Risk")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------
# Prediction Section
# ---------------------------------------------
if submit:
    with st.spinner("Analyzing patient data..."):
        input_data = np.array([[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Diabetes Detected**")
       
    else:
        st.success("✅ **Low Risk of Diabetes**")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown(
    """
    <div class="footer">
        © 2025 | KWIZERA Elisee| RP Tumba College
    </div>
    """,
    unsafe_allow_html=True
)
