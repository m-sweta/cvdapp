import streamlit as st
import numpy as np
import os
from PIL import Image
from deepface import DeepFace
import final_integrated  # Import final_integrated.py for image-related functions
import ascvd  # Import ascvd.py for ASCVD risk calculation

def upload_image():
    """Step 1: Upload and display the image."""
    st.header("Step 1: Upload Your Image")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image to calculate the risk score
        image_path = uploaded_image.name  # Save to temporary file if necessary
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Call the image prediction function from final_integrated.py
        image_risk_score = final_integrated.predict_image(image_path, classifier=None, scaler=None)  # Assuming the classifier and scaler are loaded internally
        
        if image_risk_score is None:
            st.warning("Unable to determine image-based risk score.")
        else:
            # Store the risk score in session state
            st.session_state.image_risk_score = image_risk_score
            st.session_state.image_uploaded = True
            st.write(f"Image-based Risk Score: {image_risk_score}")
    else:
        st.session_state.image_uploaded = False
        st.warning("Please upload an image to proceed.")

def clinical_inputs():
    """Step 2: Input clinical data (RCRI, STS)."""
    st.header("Step 2: Input Clinical Data")

    # Clinical Inputs
    age = st.number_input("Age", min_value=18, max_value=120, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, step=1)
    hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=10, max_value=100, step=1)
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, step=1)
    bp_treatment = st.selectbox("On Blood Pressure Medication?", ["Yes", "No"])
    smoker = st.selectbox("Do you smoke?", ["Yes", "No"])
    diabetic = st.selectbox("Do you have diabetes?", ["Yes", "No"])
    
    # RCRI and STS inputs
    rcri = st.number_input("RCRI Score", min_value=0, max_value=10, step=1)
    sts = st.number_input("STS Score", min_value=0, max_value=100, step=1)

    # Convert input for blood pressure treatment
    bp_treatment = bp_treatment == "Yes"
    smoker = smoker == "Yes"
    diabetic = diabetic == "Yes"

    # Store the clinical data in session state
    st.session_state.clinical_data = {
        "age": age,
        "sex": sex,
        "total_chol": total_chol,
        "hdl": hdl,
        "systolic_bp": systolic_bp,
        "bp_treatment": bp_treatment,
        "smoker": smoker,
        "diabetic": diabetic,
        "rcri": rcri,
        "sts": sts
    }

def calculate_risk():
