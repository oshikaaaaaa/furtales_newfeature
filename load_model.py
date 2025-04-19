
import os
import joblib
import logging

def load_model():
    """Load the saved model, features, and label encoders with correct path handling"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("model_loader")
        
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create paths to the model files using the script's directory
        model_path = os.path.join(current_dir, 'cat_health_model .joblib')
        encoders_path = os.path.join(current_dir, 'cat_label_encoders.joblib')
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None, None, None, False
            
        if not os.path.exists(encoders_path):
            logger.error(f"Encoders file not found at {encoders_path}")
            return None, None, None, False
        
        # Load the files with the correct paths
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        
        logger.info(f"Loading encoders from {encoders_path}")
        label_encoders = joblib.load(encoders_path)
        
        # Rest of validation code...
        # [...]
        
        return model_data['model'], model_data['features'], label_encoders, True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None, False
load_model()