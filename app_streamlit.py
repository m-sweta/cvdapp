import streamlit as st
import tempfile
import joblib
from model_utils import predict_image
from ascvd import calculate_ascvd_risk  # Make sure to import the ASCVD function

st.set_page_config(page_title="CVD Risk Estimator", layout="centered")
st.title("🫀 CVD Risk Estimator from Facial Photo")

uploaded_file = st.file_uploader("Upload a patient photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Run CVD Prediction"):
        with st.spinner("Analyzing photo..."):

            # Load models
            aging_classifier = joblib.load("aging_spots_classifier_model.pkl")
            aging_scaler = joblib.load("aging_spots_classifier_model_scaler.pkl")
            jugular_classifier = joblib.load("jugular_veins_classifier_model.pkl")
            jugular_scaler = joblib.load("jugular_veins_classifier_model_scaler.pkl")
            xanthelasma_classifier = joblib.load("xanthelasma_classifier_model.pkl")
            xanthelasma_scaler = joblib.load("xanthelasma_classifier_model_scaler.pkl")
            neck_classifier = joblib.load("neck_circumference_classifier_model.pkl")
            neck_scaler = joblib.load("neck_circumference_classifier_model_scaler.pkl")

            # Predict each condition
            aging_pred = predict_image(tmp_path, aging_classifier, aging_scaler)
            jugular_pred = predict_image(tmp_path, jugular_classifier, jugular_scaler)
            xanthelasma_pred = predict_image(tmp_path, xanthelasma_classifier, xanthelasma_scaler)
            neck_pred = predict_image(tmp_path, neck_classifier, neck_scaler)

            st.subheader("Predictions")
            st.write(f"Aging Spots: {aging_pred or 'Unable to determine'}")
            st.write(f"Jugular Veins: {jugular_pred or 'Unable to determine'}")
            st.write(f"Xanthelasma: {xanthelasma_pred or 'Unable to determine'}")
            st.write(f"Neck Circumference: {neck_pred or 'Unable to determine'}")

            # Score the image
            image_points = 0
            if aging_pred and aging_pred.lower() == "positive":
                image_points += 2
            if jugular_pred and jugular_pred.lower() == "positive":
                image_points += 1
            if xanthelasma_pred and xanthelasma_pred.lower() == "positive":
                image_points += 3
            if neck_pred and neck_pred.lower() == "positive":
                image_points += 2

            st.success(f"🧠 Image-Based Risk Score: {image_points}")

            # ----------------------------
            # Patient Data Input (ASCVD Risk Calculation)
            # ----------------------------
            st.subheader("Enter Patient Data for ASCVD Risk Calculation")

            # Gather patient data interactively via Streamlit widgets
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            sex = st.radio("Sex", options=["Male", "Female"], index=0)
            total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, value=200.0)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=0.0, value=50.0)
            systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=0.0, value=120.0)
            bp_treatment = st.radio("On Blood Pressure Medication?", options=["Yes", "No"], index=0)
            smoker = st.radio("Is the patient a smoker?", options=["Yes", "No"], index=0)
            diabetic = st.radio("Does the patient have diabetes?", options=["Yes", "No"], index=0)
            rcri = st.number_input("Enter RCRI score (leave blank if not available)", value=0.0)
            sts = st.number_input("Enter STS score (leave blank if not available)", value=0.0)

            # Convert inputs into the expected format for ASCVD calculation
            bp_treatment = True if bp_treatment == "Yes" else False
            smoker = True if smoker == "Yes" else False
            diabetic = True if diabetic == "Yes" else False

            # Combine the patient data into the expected format for ASCVD calculation
            patient_data = (age, sex.lower(), total_chol, hdl, systolic_bp, bp_treatment, smoker, diabetic, rcri, sts)

            if st.button("Calculate ASCVD Risk"):
                # Calculate ASCVD risk based on patient data
                ascvd_risk = calculate_ascvd_risk(*patient_data)
                final_risk = ascvd_risk + image_points * 2  # Combine with image risk points
                st.write(f"\nASCVD 10-year risk from patient data: {ascvd_risk}%")
                st.write(f"Final estimated risk of CVD: {final_risk}%")

                # Categorize the risk
                if final_risk <= 5:
                    st.write("Risk Category: Low Risk")
                elif 5 < final_risk <= 7.4:
                    st.write("Risk Category: Medium-low Risk")
                elif 7.5 <= final_risk <= 19.9:
                    st.write("Risk Category: Medium-high Risk")
                else:
                    st.write("Risk Category: High Risk")

