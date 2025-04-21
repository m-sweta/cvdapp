import streamlit as st
import tempfile
import joblib
from model_utils import predict_image
from ascvd import calculate_ascvd_risk

st.set_page_config(page_title="CVD Risk Estimator", layout="centered")
st.title("🫀 CVD Risk Estimator from Facial Photo and Patient Data")

# Session state to store intermediate results
if 'image_points' not in st.session_state:
    st.session_state.image_points = None
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'img_path' not in st.session_state:
    st.session_state.img_path = None

# Step 1: Upload image
uploaded_file = st.file_uploader("📷 Upload a patient photo", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        st.session_state.img_path = tmp.name
        st.session_state.uploaded = True
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# Step 2: Run Image-Based Risk Scoring
if st.session_state.uploaded and st.button("🔬 Analyze Facial Features"):
    with st.spinner("Running image-based prediction..."):
        # Load models
        aging_classifier = joblib.load("aging_spots_classifier_model.pkl")
        aging_scaler = joblib.load("aging_spots_classifier_model_scaler.pkl")
        jugular_classifier = joblib.load("jugular_veins_classifier_model.pkl")
        jugular_scaler = joblib.load("jugular_veins_classifier_model_scaler.pkl")
        xanthelasma_classifier = joblib.load("xanthelasma_classifier_model.pkl")
        xanthelasma_scaler = joblib.load("xanthelasma_classifier_model_scaler.pkl")
        neck_classifier = joblib.load("neck_circumference_classifier_model.pkl")
        neck_scaler = joblib.load("neck_circumference_classifier_model_scaler.pkl")

        # Predict
        aging_pred = predict_image(st.session_state.img_path, aging_classifier, aging_scaler)
        jugular_pred = predict_image(st.session_state.img_path, jugular_classifier, jugular_scaler)
        xanthelasma_pred = predict_image(st.session_state.img_path, xanthelasma_classifier, xanthelasma_scaler)
        neck_pred = predict_image(st.session_state.img_path, neck_classifier, neck_scaler)

        # Image score
        score = 0
        if aging_pred and aging_pred.lower() == "positive":
            score += 2
        if jugular_pred and jugular_pred.lower() == "positive":
            score += 1
        if xanthelasma_pred and xanthelasma_pred.lower() == "positive":
            score += 3
        if neck_pred and neck_pred.lower() == "positive":
            score += 2

        st.session_state.image_points = score

        st.subheader("📸 Image-Based Features")
        st.write(f"Aging Spots: {aging_pred or 'Unable to determine'}")
        st.write(f"Jugular Veins: {jugular_pred or 'Unable to determine'}")
        st.write(f"Xanthelasma: {xanthelasma_pred or 'Unable to determine'}")
        st.write(f"Neck Circumference: {neck_pred or 'Unable to determine'}")
        st.success(f"🧠 Image-Based Risk Score: {score}")

# Step 3: Show Patient Data Form
if st.session_state.image_points is not None:
    st.markdown("---")
    st.header("🧾 Enter Patient Information")

    with st.form("patient_data_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120)
            total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0)
            systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=0.0)
            rcri = st.number_input("RCRI Score (Optional)", min_value=0.0, step=0.1)
        with col2:
            sex = st.selectbox("Sex", ["Select", "male", "female"])
            hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=0.0)
            bp_treatment = st.radio("On BP meds?", ["Yes", "No"])
            smoker = st.radio("Smoker?", ["Yes", "No"])
            diabetic = st.radio("Diabetic?", ["Yes", "No"])
            sts = st.number_input("STS Score (Optional)", min_value=0.0, step=0.1)

        submitted = st.form_submit_button("🧮 Calculate Final CVD Risk")

    if submitted:
        with st.spinner("Calculating risk estimate..."):
            bp_treatment_bool = bp_treatment == "Yes"
            smoker_bool = smoker == "Yes"
            diabetic_bool = diabetic == "Yes"

            try:
                ascvd_risk = calculate_ascvd_risk(
                    age,
                    sex if sex != "Select" else None,
                    total_chol,
                    hdl,
                    systolic_bp,
                    bp_treatment_bool,
                    smoker_bool,
                    diabetic_bool,
                    rcri if rcri > 0 else None,
                    sts if sts > 0 else None
                )
            except Exception as e:
                ascvd_risk = 0
                st.error(f"ASCVD Risk Calculation Failed: {e}")

            final_risk = ascvd_risk + st.session_state.image_points * 2

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
