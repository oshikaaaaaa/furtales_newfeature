from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_loader")

# Load model and encoders once
def load_model():
    """Load the saved model, features, and label encoders."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'cat_health_model.joblib')
        encoders_path = os.path.join(current_dir, 'cat_label_encoders.joblib')

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None, None, None, False
        if not os.path.exists(encoders_path):
            logger.error(f"Encoders file not found at {encoders_path}")
            return None, None, None, False
        
        model_data = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        
        return model_data['model'], model_data['features'], label_encoders, True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None, False

# Loading model and encoders at the beginning
model, model_features, label_encoders, success = load_model()

# FastAPI setup
app = FastAPI()

# Define the Input data structure
class InputData(BaseModel):
    food_midpoint: float  # Example preprocessed feature
    sleep_estimate_encoded: int
    mood_encoded: int
    activity_level_encoded: int
    vocalization_level_encoded: int
    affection_level_encoded: int

# Function to analyze the input data and make prediction
def analyze_new_cat_log(log_dict, model, model_features, label_encoders):
    # Assuming log_dict contains preprocessed features
    input_values = [
        log_dict.get("food_midpoint"),
        log_dict.get("sleep_estimate_encoded"),
        log_dict.get("mood_encoded"),
        log_dict.get("activity_level_encoded"),
        log_dict.get("vocalization_level_encoded"),
        log_dict.get("affection_level_encoded")
    ]
    
    # Make prediction
    prediction = model.predict([input_values])[0]
    return prediction

@app.post("/predict")
def predict(data: InputData):
    # Convert the InputData into a dictionary for the analysis function
    log_dict = data.dict()

    # Use the `analyze_new_cat_log` function to process the data and get the prediction
    prediction = analyze_new_cat_log(log_dict, model, model_features, label_encoders)
    return {"prediction": prediction}
