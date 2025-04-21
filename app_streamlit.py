import streamlit as st
import tempfile
import joblib
from model_utils import predict_image
from final_integrated import calculate_combined_risk

st.set_page_config(page_title="CVD Risk Estimator", layout="centered")
st.title("🫠 CVD Risk Estimator from Facial Photo")

# --- Step 1: Upload Image ---
st.header("📁 Step 1: Upload Image")
uploaded_file = st.file_uploader("Upload a patient photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.session_state["uploaded_image_path"] = tmp_path
    st.session_state["image_uploaded"] = True

# --- Step 2: Analyze Image ---
if st.session_state.get("image_uploaded"):
    st.markdown("---")
    st.header("🧠 Step 2: Analyze Facial Markers")

    if st.button("🔍 Run Image-Based CVD Prediction"):
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
            tmp_path = st.session_state["uploaded_image_path"]
            aging_pred = predict_image(tmp_path, aging_classifier, aging_scaler)
            jugular_pred = predict_image(tmp_path, jugular_classifier, jugular_scaler)
            xanthelasma_pred = predict_image(tmp_path, xanthelasma_classifier, xanthelasma_scaler)
            neck_pred = predict_image(tmp_path, neck_classifier, neck_scaler)

            st.subheader("Image-Based Predictions")
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

            st.session_state["image_score_ready"] = True
            st.session_state["image_points"] = image_points

            st.success(f"✅ Image-Based Risk Score: {image_points}")

# --- Step 3: Clinical Form and Final Risk Calculation ---
st.markdown("---")
st.header("📊 Step 3: Enter Clinical Info and Calculate Final Risk")

if st.session_state.get("image_score_ready"):
    with st.form("clinical_data_form"):
        st.text_input("🔍 Image-Based Risk Score (from Step 2)", value=str(st.session_state.image_points), disabled=True)

        age = st.number_input("Age", min_value=18, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
        total_cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        hdl_cholesterol = st.number_input("HDL Cholesterol (mg/dL)", min_value=10, max_value=100, value=50)
        smoker = st.checkbox("Current Smoker?")
        diabetes = st.checkbox("Has Diabetes?")
        history_cad = st.checkbox("History of Coronary Artery Disease?")
        recent_mi = st.checkbox("Recent Myocardial Infarction?")
        rcri = st.number_input("RCRI Score", min_value=0, max_value=6, value=0)
        sts_score = st.number_input("STS Score (%)", min_value=0.0, max_value=100.0, value=1.0)

        submitted = st.form_submit_button("✅ Calculate Final CVD Risk")
        if submitted:
            image_score = st.session_state.get("image_points", 0)
            final_risk = calculate_combined_risk(
                age=age,
                gender=gender,
                systolic_bp=systolic_bp,
                total_cholesterol=total_cholesterol,
                hdl_cholesterol=hdl_cholesterol,
                smoker=smoker,
                diabetes=diabetes,
                history_cad=history_cad,
                recent_mi=recent_mi,
                rcri=rcri,
                sts_score=sts_score,
                image_score=image_score
            )
            st.success(f"🧥 Final Combined CVD Risk: **{final_risk:.2f}%**")
else:
    st.warning("Please complete the image analysis step first to continue.")
