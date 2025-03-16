import torch
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Union
import os
import tempfile
from pathlib import Path

from app.settings.config import settings
from app.utils.google_drive import download_file_from_google_drive

# Model and preprocessing objects
model = None
scaler = None

# File IDs from your Google Drive links
MODEL_WEIGHTS_ID = "1Y2RXeA9dq5D3j1Ph7QP-QnoIDHQxXC4c"
SCALER_ID = "1xonF91VuOsQ2ZmkeLrRpLuhhc3Ey69WR"
MODEL_INFO_ID = "1xonF91VuOsQ2ZmkeLrRpLuhhc3Ey69WR"  # You provided the same link for both .pkl and .json

class DirectNonLinearNN(torch.nn.Module):
    """DON prediction neural network model"""
    def __init__(self, input_dim=447, output_range=12.0):
        super(DirectNonLinearNN, self).__init__()
        # A single linear layer
        self.fc = torch.nn.Linear(input_dim, 1)
        self.output_range = output_range
    
    def forward(self, x):
        # Apply linear transformation
        linear_output = self.fc(x)
        
        # Apply sigmoid and scale
        non_linear_output = torch.sigmoid(linear_output) * self.output_range
        
        return non_linear_output

def load_model():
    """
    Load the saved model and preprocessing components for inference from Google Drive
    """
    global model, scaler
    
    # If already loaded, return
    if model is not None and scaler is not None:
        return
    
    # Create a temporary directory for model files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for model files: {temp_dir}")
    
    # Define local file paths
    local_model_info_path = os.path.join(temp_dir, "model_info.json")
    local_model_path = os.path.join(temp_dir, "model.pt")
    local_scaler_path = os.path.join(temp_dir, "scaler.pkl")
    
    try:
        # Download model info
        print(f"Downloading model info file from Google Drive (ID: {MODEL_INFO_ID})...")
        download_file_from_google_drive(MODEL_INFO_ID, local_model_info_path)
        print("Model info file downloaded successfully")
        
        # Load model architecture info
        with open(local_model_info_path, "r") as f:
            model_info = json.load(f)
        
        # Create model with same architecture
        model = DirectNonLinearNN(
            input_dim=model_info.get("input_dim", 447),
            output_range=model_info.get("output_range", 12.0)
        )
        
        # Download and load model weights
        print(f"Downloading model weights from Google Drive (ID: {MODEL_WEIGHTS_ID})...")
        download_file_from_google_drive(MODEL_WEIGHTS_ID, local_model_path)
        print("Model weights downloaded successfully")
        
        model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        
        # Download and load scaler
        print(f"Downloading scaler from Google Drive (ID: {SCALER_ID})...")
        download_file_from_google_drive(SCALER_ID, local_scaler_path)
        print("Scaler downloaded successfully")
        
        with open(local_scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        print("Model and preprocessing components loaded successfully from Google Drive")
    
    except Exception as e:
        error_msg = f"Error loading model from Google Drive: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def predict_don(features: List[float]) -> float:
    """
    Make DON prediction using the loaded model
    
    Args:
        features: List of feature values in the correct order
        
    Returns:
        float: The predicted DON value
    """
    global model, scaler
    
    # Load model if not already loaded
    if model is None or scaler is None:
        load_model()
    
    # Convert features to numpy array
    input_features = np.array([features])
    
    # Apply preprocessing
    X_scaled = scaler.transform(input_features)
    
    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        log_predictions = model(X_tensor)
    
    # Convert back to numpy
    log_predictions = log_predictions.numpy().flatten()
    
    # Convert from log scale to original scale
    predictions = np.expm1(log_predictions)
    
    # Ensure no negative values
    predictions = np.maximum(predictions, 0)
    
    return float(predictions[0])