import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
from deepface import DeepFace
import ascvd

# ----------------------------
# Helper Functions for Image Processing
# ----------------------------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Loads an image, converts it to RGB, resizes, and ensures it has 3 channels."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def extract_embedding(image):
    """Extracts the VGG-Face embedding from an image using DeepFace."""
    result = DeepFace.represent(img_path=image, model_name="VGG-Face", enforce_detection=False)
    if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
        return result[0]["embedding"]
    return None

def predict_image(image_path, classifier, scaler, target_size=(224, 224)):
    """Processes an image and returns the predicted class using the provided classifier and scaler."""
    img = load_and_preprocess_image(image_path, target_size)
    if img is None:
        return None
    embedding = extract_embedding(img)
    if embedding is None:
        return None
    embedding = np.array(embedding).reshape(1, -1)
    embedding_scaled = scaler.transform(embedding)
    prediction = classifier.predict(embedding_scaled)
    return prediction[0]

# ----------------------------
# Step 1: Image Upload
# ----------------------------
def upload_image():
    """Step 1: Upload the image and calculate the image-based risk score."""
    st.header("Step 1: Upload Image")
    uploaded_image = st.file_uploader("Upload an image (e.g., face, neck, etc.)", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        image_path = f"temp_image.{uploaded_image.name.split('.')[-1]}"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Set image_uploaded flag and store image risk score in session state
        st.session_state.image_uploaded = True
        image_risk_score = calculate_image_risk(image_path)
        st.session_state.image_risk_score = image_risk_score
        st.write(f"Image-based risk score: {image_risk_score}")
    else:
        st.session_state.image_uploaded = False

# ----------------------------
# Step 2: Clinical Data Input
# ----------------------------
def enter_clinical_data():
    """Step 2: Enter clinical data (age, cholesterol, BP, etc.)."""
    st.header("Step 2: Enter Clinical Data")

    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0, max_value=300, value=200)
    hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=0, max_value=100, value=50)
    systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
    bp_treatment = st.selectbox("On blood pressure medication?", ["Yes", "No"])
    smoker = st.selectbox("Is the patient a smoker?", ["Yes", "No"])
    diabetic = st.selectbox("Does the patient have diabetes?", ["Yes", "No"])

    rcri = st.number_input("Enter RCRI score (if available)", min_value=0, max_value=10, value=0)
    sts = st.number_input("Enter STS score (if available)", min_value=0, max_value=10, value=0)

    clinical_data = {
        "age": age,
        "sex": sex.lower(),
        "total_chol": total_chol,
        "hdl": hdl,
        "systolic_bp": systolic_bp,
        "bp_treatment": True if bp_treatment == "Yes" else False,
        "smoker": True if smoker == "Yes" else False,
        "diabetic": True if diabetic == "Yes" else False,
        "rcri": rcri,
        "sts": sts
    }

    # Store clinical data in session state
    st.session_state.clinical_data = clinical_data

# ----------------------------
# Step 3: Calculate Final CVD Risk
# ----------------------------
def calculate_risk():
    """Step 3: Calculate and display the final CVD risk."""
    st.header("Step 3: Calculate Final CVD Risk")

    if st.session_state.image_uploaded:
        # Get image-based risk score from session state
        image_risk_score = st.session_state.image_risk_score
    else:
        image_risk_score = 0  # Default to 0 if no image uploaded

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

    # Calculate base ASCVD risk using clinical data from ascvd.py
    base_ascvd_risk = ascvd.calculate_ascvd_risk(
        age, sex, total_chol, hdl, systolic_bp, bp_treatment, smoker, diabetic, rcri, sts
    )
    
    # Combine image-based risk score if available
    final_risk = base_ascvd_risk + image_risk_score * 2  # Arbitrary weight for image score, adjust as needed

    # Store final risk in session state
    st.session_state.final_risk = final_risk
    st.write(f"Final CVD Risk: {final_risk:.2f}%")
    
    # Final button to lock in the risk
    if st.button("Lock Final CVD Risk"):
        st.success("The final CVD risk has been locked.")

# ----------------------------
# Main Function
# ----------------------------
def main():
    """Run the Streamlit app."""
    st.title("CVD Risk Estimator")
    
    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False
    if "final_risk" not in st.session_state:
        st.session_state.final_risk = None

    # Step 1: Upload image
    upload_image()

    # Step 2: Enter clinical data
    enter_clinical_data()

    # Step 3: Calculate final risk
    calculate_risk()

if __name__ == "__main__":
    main()

