import streamlit as st
import cv2
import numpy as np
from PIL import Image
import final_integrated  # Make sure this is the correct import
import ascvd  # Ensure ascvd.py is available in the same directory or imported properly

def upload_image():
    """Step 1: Upload and display the image."""
    st.header("Step 1: Upload Your Image")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image to calculate the risk score
        img = Image.open(uploaded_image)
        image_array = np.array(img)
        risk_score = final_integrated.calculate_image_based_risk(image_array)  # Call the relevant function

        # Store the risk score in session state
        st.session_state.image_risk_score = risk_score
        st.session_state.image_uploaded = True
        st.write(f"Image-based Risk Score: {risk_score}%")
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
    """Step 3: Calculate and display the final CVD risk."""
    st.header("Step 3: Calculate Final CVD Risk")

    if st.session_state.image_uploaded:
        # Get image-based risk score from session state
        image_risk_score = st.session_state.image_risk_score
    else:
        image_risk_score = None

    # Get clinical data from session state
    clinical_data = st.session_state.clinical_data
    age = clinical_data["age"]
    sex = clinical_data["sex"]
    total_chol = clinical_data["total_chol"]
    hdl = clinical_data["hdl"]
    systolic_bp = clinical_data["systolic_bp"]
    bp_treatment = clinical_data["bp_treatment"]
    smoker = clinical_data["smoker"]
    diabetic = clinical_data["diabetic"]
    rcri = clinical_data["rcri"]
    sts = clinical_data["sts"]

    # Calculate base ASCVD risk using clinical data
    base_ascvd_risk = ascvd.calculate_ascvd_risk(
        age, sex, total_chol, hdl, systolic_bp, bp_treatment, smoker, diabetic, rcri, sts
    )
    
    # Combine image-based risk score if available
    if image_risk_score is not None:
        final_risk = (base_ascvd_risk + image_risk_score) / 2
    else:
        final_risk = base_ascvd_risk

    # Lock the final risk score into the final form
    st.session_state.final_risk = final_risk
    st.write(f"Final CVD Risk: {final_risk:.2f}%")
    
    # Final button to lock in the risk
    if st.button("Lock Final CVD Risk"):
        st.success("The final CVD risk has been locked.")

def main():
    """Run the Streamlit app."""
    st.title("CVD Risk Calculator")

    # Initialize session state variables
    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False
    if "image_risk_score" not in st.session_state:
        st.session_state.image_risk_score = None
    if "clinical_data" not in st.session_state:
        st.session_state.clinical_data = {}
    if "final_risk" not in st.session_state:
        st.session_state.final_risk = None

    # Step 1: Upload Image
    upload_image()

    # Step 2: Input Clinical Data
    clinical_inputs()

    # Step 3: Calculate Final CVD Risk
    calculate_risk()

if __name__ == "__main__":
    main()

