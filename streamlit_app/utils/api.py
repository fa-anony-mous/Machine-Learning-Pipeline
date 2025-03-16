import requests
import os
import json
from typing import List, Dict, Any, Optional, Tuple

def get_api_url() -> str:
    """
    Get the API URL based on environment
    
    Returns:
        str: The API base URL
    """
    # In production (Vercel), API is at the same domain under /api
    # In development, it's at the FastAPI server URL
    if os.environ.get("VERCEL_ENV") in ["production", "preview"]:
        return ""
    else:
        return " http://127.0.0.1:8000"

def make_prediction(features: List[float]) -> Tuple[bool, Any]:
    """
    Make a DON prediction by calling the API
    
    Args:
        features: List of feature values
        
    Returns:
        tuple: (success: bool, result: dict or error message)
    """
    api_url = get_api_url()
    
    try:
        response = requests.post(
            f"{api_url}/predictions/",
            json={"features": features},
            timeout=10
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Request error: {str(e)}"

def get_prediction_by_id(prediction_id: str) -> Tuple[bool, Any]:
    """
    Get a prediction by ID
    
    Args:
        prediction_id: ID of the prediction
        
    Returns:
        tuple: (success: bool, result: dict or error message)
    """
    api_url = get_api_url()
    
    try:
        response = requests.get(
            f"{api_url}/predictions/{prediction_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Request error: {str(e)}"

def get_recent_predictions(limit: int = 10) -> Tuple[bool, Any]:
    """
    Get recent predictions
    
    Args:
        limit: Maximum number of predictions to return
        
    Returns:
        tuple: (success: bool, result: list or error message)
    """
    api_url = get_api_url()
    
    try:
        response = requests.get(
            f"{api_url}/predictions/?limit={limit}",
            timeout=10
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Request error: {str(e)}"

def check_api_health() -> Tuple[bool, str]:
    """
    Check if the API is healthy
    
    Returns:
        tuple: (success: bool, message: str)
    """
    api_url = get_api_url()
    
    try:
        response = requests.get(
            f"{api_url}/health",
            timeout=5
        )
        
        if response.status_code == 200:
            return True, "API is healthy"
        else:
            return False, f"API returned status code {response.status_code}"
    except Exception as e:
        return False, f"API connection failed: {str(e)}"