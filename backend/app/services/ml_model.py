import torch
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Union
import os
from pathlib import Path

from app.settings.config import settings

# Model and preprocessing objects
model = None
scaler = None

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
    Load the saved model and preprocessing components for inference
    """
    global model, scaler
    
    # If already loaded, return
    if model is not None and scaler is not None:
        return
    
    # Ensure the model directory exists
    if not os.path.exists(settings.MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {settings.MODEL_DIR}")
    
    # Load model architecture info
    try:
        with open(settings.MODEL_INFO_PATH, "r") as f:
            model_info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model info file not found: {settings.MODEL_INFO_PATH}")
    
    # Create model with same architecture
    model = DirectNonLinearNN(
        input_dim=model_info.get("input_dim", 447),
        output_range=model_info.get("output_range", 12.0)
    )
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
    except FileNotFoundError:
        raise FileNotFoundError(f"Model weights file not found: {settings.MODEL_PATH}")
    
    # Load scaler
    try:
        with open(settings.SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file not found: {settings.SCALER_PATH}")
    
    print("Model and preprocessing components loaded successfully")

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