import os
import sys
import argparse
import cv2
import joblib
import numpy as np
from deepface import DeepFace
from ascvd import calculate_ascvd_risk  # Assuming ascvd.py is in the same directory or PYTHONPATH

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
# Main Function
# ----------------------------
def main():
    # Define the image path
    image_path = input("Enter the path to the image file: ")

    # Ensure image path is valid
    if not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist.")
        return
    
    target_size = (224, 224)

    # Load pre-trained models and scalers
    aging_classifier = joblib.load("aging_spots_classifier_model.pkl")
    aging_scaler = joblib.load("aging_spots_classifier_model_scaler.pkl")
    jugular_classifier = joblib.load("jugular_veins_classifier_model.pkl")
    jugular_scaler = joblib.load("jugular_veins_classifier_model_scaler.pkl")
    xanthelasma_classifier = joblib.load("xanthelasma_classifier_model.pkl")
    xanthelasma_scaler = joblib.load("xanthelasma_classifier_model_scaler.pkl")
    neck_classifier = joblib.load("neck_circumference_classifier_model.pkl")
    neck_scaler = joblib.load("neck_circumference_classifier_model_scaler.pkl")

    # Run image through each model and get predictions
    print("\nRunning image through each model...\n")
    
    aging_pred = predict_image(image_path, aging_classifier, aging_scaler, target_size=target_size)
    jugular_pred = predict_image(image_path, jugular_classifier, jugular_scaler, target_size=target_size)
    xanthelasma_pred = predict_image(image_path, xanthelasma_classifier, xanthelasma_scaler, target_size=target_size)
    neck_pred = predict_image(image_path, neck_classifier, neck_scaler, target_size=target_size)
    
    # Print predictions for each model
    print(f"Aging Spots Prediction: {aging_pred if aging_pred else 'Unable to determine'}")
    print(f"Jugular Veins Prediction: {jugular_pred if jugular_pred else 'Unable to determine'}")
    print(f"Xanthelasma Prediction: {xanthelasma_pred if xanthelasma_pred else 'Unable to determine'}")
    print(f"Neck Circumference Prediction: {neck_pred if neck_pred else 'Unable to determine'}")
    
    # Calculate image-based risk points (customize these logic as necessary)
    image_points = 0
    if aging_pred and aging_pred.lower() == "positive":
        image_points += 2
    if jugular_pred and jugular_pred.lower() == "positive":
        image_points += 1
    if xanthelasma_pred and xanthelasma_pred.lower() == "positive":
        image_points += 3
    if neck_pred and neck_pred.lower() == "positive":
        image_points += 2

    print(f"\nImage-based risk points: {image_points}")
    
    # You can integrate additional patient data or ASCVD calculation here if needed.
    # Assuming get_patient_data() and calculate_ascvd_risk() is implemented as in the original code

if __name__ == "__main__":
    main()
