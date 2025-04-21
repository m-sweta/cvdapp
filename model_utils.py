# model_utils.py

import cv2
import numpy as np
from deepface import DeepFace

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def extract_embedding(image_path):
    result = DeepFace.represent(img_path=image_path, model_name="VGG-Face", enforce_detection=False)
    if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
        return result[0]["embedding"]
    return None

def predict_image(image_path, classifier, scaler, target_size=(224, 224)):
    img = load_and_preprocess_image(image_path, target_size)
    if img is None:
        return None
    embedding = extract_embedding(image_path)
    if embedding is None:
        return None
    embedding = np.array(embedding).reshape(1, -1)
    embedding_scaled = scaler.transform(embedding)
    prediction = classifier.predict(embedding_scaled)
    return prediction[0]
