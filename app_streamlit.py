import streamlit as st
import tempfile
import joblib
from model_utils import predict_image
from ascvd import calculate_ascvd_risk  # Make sure ascvd.py is in the same directory

st.set_page_config(page_title="CVD Risk Estimator", layout="centered")
st.title("🫀 CVD Risk Estimator from Facial Photo + Patient Data")

# Session state setup
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "image_points" not in st.session_state:
    st.session_state.image_points = 0
if "show_patient_form" not in st.session_state:
    st.session_state.show_patient_form = False

# Step 1: Upload photo
uploaded_file = st.file_uploader("Step 1: Upload a patient photo", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        st.session_state.image_path = tmp.name
        st.session_state.image_uploaded = True
        st.session_state.image_points = 0  # Reset

if st.session_state.image_uploaded:
    st.image(st.session_state.image_path, caption="Uploaded Image", use_container_width=True)

# Step 2: Run Image-Based Prediction
if st.session_state.image_uploaded and st.button("Step 2: Analyze Facial Markers"):
    with st.spinner("Analyzing image..."):
        aging_classifier = joblib.load("aging_spots_classifier_model.pkl")
        aging_scaler = joblib.load("aging_spots_classifier_model_scaler.pkl")
        jugular_classifier = joblib.load("jugular_veins_classifier_model.pkl")
        jugular_scaler = joblib.load("jugular_veins_classifier_model_scaler.pkl")
        xanthelasma_classifier = joblib.load("xanthelasma_classifier_model.pkl")
        xanthelasma_scaler = joblib.load("xanthelasma_classifier_model_scaler.pkl")
        neck_classifier = joblib.load("neck_circumference_classifier_model.pkl")
        neck_scaler = joblib.load("neck_circumference_classifier_model_scaler.pkl")

        # Predict
        aging_pred = predict_image(st.session_state.image_path, aging_classifier, aging_scaler)
        jugular_pred = predict_image(st.session_state.image_path, jugular_classifier, jugular_scaler)
        xanthelasma_pred = predict_image(st.session_state.image_path, xanthelasma_classifier, xanthelasma_scaler)
        neck_pred = predict_image(st.session_state.image_path, neck_classifier, neck_scaler)

        st.subheader("Model Predictions:")
        st.write(f"Aging Spots: {aging_pred or 'Unable to determine'}")
        st.write(f"Jugular Veins: {jugular_pred or 'Unable to determine'}")
        st.write(f"Xanthelasma: {xanthelasma_pred or 'Unable to determine'}")
        st.write(f"Neck Circumference: {neck_pred or 'Unable to determine'}")

        # Risk score
        image_points = 0
        if aging_pred and aging_pred.lower() == "positive":
            image_points += 2
        if jugular_pred and jugular_pred.lower() == "positive":
            image_points += 1
        if xanthelasma_pred and xanthelasma_pred.lower() == "positive":
            image_points += 3
        if neck_pred and neck_pred.lower() == "positive":
            image_points += 2

        st.session_state.image_points = image_points
        st.success(f"🧠 Image-Based Risk Score: {image_points}")
        st.session_state.show_patient_form = True

# Step 3: Patient Data Input
if st.session_state.show_patient_form:
    st.subheader("Step 3: Enter Patient Clinical Data")

    with st.form("ascvd_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, step=1)
            total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0)
            systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=0.0)
            rcri = st.number_input("RCRI Score (optional)", min_value=0.0, value=0.0)
        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            hdl = st.number_input("HDL (mg/dL)", min_value=0.0)
            bp_treatment = st.checkbox("On BP medication?")
            smoker = st.checkbox("Smoker?")
            diabetic = st.checkbox("Diabetic?")
            sts = st.number_input("STS Score (optional)", min_value=0.0, value=0.0)

        submitted = st.form_submit_button("Calculate Final CVD Risk")
        if submitted:
            ascvd_risk = calculate_ascvd_risk(
                age, sex, total_chol, hdl, systolic_bp,
                bp_treatment, smoker, diabetic, rcri or None, sts or None
            )
            final_risk = ascvd_risk + st.session_state.image_points * 2

            st.subheader("🧮 Final CVD Risk Estimate")
            st.write(f"ASCVD 10-year risk: {ascvd_risk:.2f}%")
            st.write(f"Combined Risk Score: {final_risk:.2f}%")

            if final_risk <= 5:
                category = "Low Risk"
            elif 5 < final_risk <= 7.4:
                category = "Medium-low Risk"
            elif 7.5 <= final_risk <= 19.9:
                category = "Medium-high Risk"
            else:
                category = "High Risk"
            st.success(f"🩺 Risk Category: **{category}**")

