import streamlit as st
import tempfile
import joblib
from model_utils import predict_image

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
