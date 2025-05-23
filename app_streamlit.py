import streamlit as st
import tempfile
import joblib
from model_utils import predict_image
from ascvd import calculate_ascvd_risk  # Ensure ascvd.py is in your project

st.set_page_config(page_title="CVD Risk Estimator", layout="centered")
st.title("🫀 CVD Risk Estimator from Facial Photo and Patient Data")

# File upload
uploaded_file = st.file_uploader("Upload a patient photo", type=["jpg", "jpeg", "png"])

# Display image immediately after upload
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

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

        # Image predictions
        aging_pred = predict_image(tmp_path, aging_classifier, aging_scaler)
        jugular_pred = predict_image(tmp_path, jugular_classifier, jugular_scaler)
        xanthelasma_pred = predict_image(tmp_path, xanthelasma_classifier, xanthelasma_scaler)
        neck_pred = predict_image(tmp_path, neck_classifier, neck_scaler)

        st.subheader("🔬 Image Predictions")
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

        st.success(f"🧠 Image-Based Risk Score: {image_points}")

# Patient inputs
st.header("🧾 Patient Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120)
    total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0)
    systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=0.0)
    rcri = st.number_input("RCRI Score (Optional)", min_value=0.0, step=0.1)

with col2:
    sex = st.selectbox("Sex", ["Select", "male", "female"])
    hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=0.0)
    bp_treatment = st.radio("On BP medication?", ["Yes", "No"])
    smoker = st.radio("Smoker?", ["Yes", "No"])
    diabetic = st.radio("Diabetic?", ["Yes", "No"])
    sts = st.number_input("STS Score (Optional)", min_value=0.0, step=0.1)

# Run button
if uploaded_file is not None and st.button("🔍 Run CVD Prediction"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

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

        # Image predictions
        aging_pred = predict_image(tmp_path, aging_classifier, aging_scaler)
        jugular_pred = predict_image(tmp_path, jugular_classifier, jugular_scaler)
        xanthelasma_pred = predict_image(tmp_path, xanthelasma_classifier, xanthelasma_scaler)
        neck_pred = predict_image(tmp_path, neck_classifier, neck_scaler)

        st.subheader("🔬 Image Predictions")
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

        st.success(f"🧠 Image-Based Risk Score: {image_points}")

        # Prepare inputs
        bp_treatment_bool = bp_treatment == "Yes"
        smoker_bool = smoker == "Yes"
        diabetic_bool = diabetic == "Yes"

        # Risk calculations
        try:
            ascvd_risk = calculate_ascvd_risk(
                age, sex if sex != "Select" else None, total_chol, hdl,
                systolic_bp, bp_treatment_bool, smoker_bool,
                diabetic_bool, rcri if rcri > 0 else None, sts if sts > 0 else None
            )
        except Exception as e:
            ascvd_risk = 0
            st.error(f"ASCVD Risk Calculation Failed: {e}")

        final_risk = ascvd_risk + image_points * 2

        st.subheader("📊 Final Risk Estimate")
        st.write(f"ASCVD 10-Year Risk: **{ascvd_risk:.2f}%**")
        st.write(f"Final Estimated CVD Risk: **{final_risk:.2f}%**")

        if final_risk <= 5:
            st.success("Risk Category: Low Risk")
        elif 5 < final_risk <= 7.4:
            st.info("Risk Category: Medium-low Risk")
        elif 7.5 <= final_risk <= 19.9:
            st.warning("Risk Category: Medium-high Risk")
        else:
            st.error("Risk Category: High Risk")

