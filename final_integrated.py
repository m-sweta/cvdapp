#!/usr/bin/env python3

#python final_integrated.py --test_image ~/Desktop/CVDAPPMaster/test.png

try:
    import cv2
except ModuleNotFoundError:
    print("cv2 not installed")
    
import os
import cv2
import numpy as np
import argparse
import joblib
from deepface import DeepFace
from ascvd import calculate_ascvd_risk  # Ensure ascvd.py is in the same directory or on PYTHONPATH

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
# Patient Data Input
# ----------------------------
def get_patient_data():
    """
    Prompts the user for patient data (age, sex, cholesterol, blood pressure, etc.)
    as well as optional RCRI and STS scores. Returns a tuple of all the parameters.
    """
    def get_input(prompt, type_func):
        s = input(prompt)
        return type_func(s) if s.strip() != "" else None

    print("Enter patient data (leave blank if not available):")
    age = get_input("Age: ", int)
    sex = input("Sex (male/female): ").strip().lower() or None
    if sex in ['m']:
        sex = 'male'
    elif sex in ['f']:
        sex = 'female'
    total_chol = get_input("Total Cholesterol (mg/dL): ", float)
    hdl = get_input("HDL Cholesterol (mg/dL): ", float)
    systolic_bp = get_input("Systolic Blood Pressure (mm Hg): ", float)
    
    bp_in = input("On blood pressure medication? (yes/no): ").strip().lower()
    bp_treatment = True if bp_in in ['yes', 'y'] else False if bp_in in ['no', 'n'] else None

    smoker_in = input("Is the patient a smoker? (yes/no): ").strip().lower()
    smoker = True if smoker_in in ['yes', 'y'] else False if smoker_in in ['no', 'n'] else None

    diabetic_in = input("Does the patient have diabetes? (yes/no): ").strip().lower()
    diabetic = True if diabetic_in in ['yes', 'y'] else False if diabetic_in in ['no', 'n'] else None

    rcri_input = input("Enter RCRI score (if available, otherwise leave blank): ").strip()
    sts_input = input("Enter STS score (if available, otherwise leave blank): ").strip()
    
    rcri = float(rcri_input) if rcri_input != "" else None
    sts = float(sts_input) if sts_input != "" else None
    
    return age, sex, total_chol, hdl, systolic_bp, bp_treatment, smoker, diabetic, rcri, sts

# ----------------------------
# Main Integrated Pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Integrated CVD Risk Estimator: Combines image-based symptom predictions and patient data."
    )
    parser.add_argument('--test_image', type=str, required=True, help="Path to the test image")
    parser.add_argument('--target_size', type=int, default=224, help="Target image size (default 224)")
    args = parser.parse_args()
    
    target_size = (args.target_size, args.target_size)
    
    # Load pre-trained models and scalers for all four models.
    aging_classifier = joblib.load("aging_spots_classifier_model.pkl")
    aging_scaler = joblib.load("aging_spots_classifier_model_scaler.pkl")
    jugular_classifier = joblib.load("jugular_veins_classifier_model.pkl")
    jugular_scaler = joblib.load("jugular_veins_classifier_model_scaler.pkl")
    xanthelasma_classifier = joblib.load("xanthelasma_classifier_model.pkl")
    xanthelasma_scaler = joblib.load("xanthelasma_classifier_model_scaler.pkl")
    neck_classifier = joblib.load("neck_circumference_classifier_model.pkl")
    neck_scaler = joblib.load("neck_circumference_classifier_model_scaler.pkl")
    
    print("\nRunning image through each model...\n")
    
    aging_pred = predict_image(args.test_image, aging_classifier, aging_scaler, target_size=target_size)
    jugular_pred = predict_image(args.test_image, jugular_classifier, jugular_scaler, target_size=target_size)
    xanthelasma_pred = predict_image(args.test_image, xanthelasma_classifier, xanthelasma_scaler, target_size=target_size)
    neck_pred = predict_image(args.test_image, neck_classifier, neck_scaler, target_size=target_size)
    
    # Print predictions for each model.
    if aging_pred is not None:
        print(f"Aging Spots Prediction: {aging_pred}")
    else:
        print("Aging Spots Prediction: Unable to determine")
        
    if jugular_pred is not None:
        print(f"Jugular Veins Prediction: {jugular_pred}")
    else:
        print("Jugular Veins Prediction: Unable to determine")
        
    if xanthelasma_pred is not None:
        print(f"Xanthelasma Prediction: {xanthelasma_pred}")
    else:
        print("Xanthelasma Prediction: Unable to determine")
        
    if neck_pred is not None:
        print(f"Neck Circumference Prediction: {neck_pred}")
    else:
        print("Neck Circumference Prediction: Unable to determine")
    
    # Assign image-based risk points based on predictions.
    image_points = 0
    # (Point values are arbitrary; adjust as needed.)
    if aging_pred is not None and aging_pred.lower() == "positive":
        image_points += 2
    if jugular_pred is not None and jugular_pred.lower() == "positive":
        image_points += 1
    if xanthelasma_pred is not None and xanthelasma_pred.lower() == "positive":
        image_points += 3
    if neck_pred is not None and neck_pred.lower() == "positive":
        image_points += 2  # Assign 2 points for a positive neck circumference result
    
    print(f"\nImage-based risk points: {image_points}")
    
    # Gather patient data
    patient_data = get_patient_data()
    # Unpack patient data into expected parameters for the ASCVD risk calculator
    # (Assuming the function expects: age, sex, total_chol, hdl, systolic_bp, bp_treatment, smoker, diabetic, rcri, sts)
    ascvd_risk = calculate_ascvd_risk(*patient_data)
    print(f"\nASCVD 10-year risk from patient data: {ascvd_risk}%")
    
    # Combine risks (for example, image points are weighted by 2, then added to ASCVD risk)
    final_risk = ascvd_risk + image_points * 2
    print(f"\nFinal estimated risk of CVD: {final_risk}%")
    
    # Optional risk categorization
    if final_risk <= 5:
        print("Risk Category: Low Risk")
    elif 5 < final_risk <= 7.4:
        print("Risk Category: Medium-low Risk")
    elif 7.5 <= final_risk <= 19.9:
        print("Risk Category: Medium-high Risk")
    else:  # final_risk >= 20
        print("Risk Category: High Risk")

if __name__ == "__main__":
    main()
