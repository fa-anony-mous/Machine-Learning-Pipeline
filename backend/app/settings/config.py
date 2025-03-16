import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directory of the backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Parent directory containing both frontend and backend
PROJECT_ROOT = BASE_DIR.parent

# Path to model artifacts (local, but we'll use cloud URLs instead)
MODEL_DIR = os.path.join(BASE_DIR, "app", "resources", "model_artifacts")

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    API_TITLE: str = "DON Prediction API"
    API_DESCRIPTION: str = "API for Deoxynivalenol (DON) prediction"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # MongoDB settings
    MONGO_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "don_predictions"
    
    # Model settings - Local paths (not used with cloud storage approach)
    MODEL_DIR: str = MODEL_DIR
    MODEL_PATH: str = os.path.join(MODEL_DIR, "don_prediction_model.pt")
    SCALER_PATH: str = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    MODEL_INFO_PATH: str = os.path.join(MODEL_DIR, "model_info.json")
    
    # Model URLs for cloud storage (replace these with your actual URLs)
    MODEL_INFO_URL: str = "https://drive.google.com/file/d/1Y2RXeA9dq5D3j1Ph7QP-QnoIDHQxXC4c/view?usp=drive_link"
    MODEL_WEIGHTS_URL: str = "https://drive.google.com/uc?export=download&id=YOUR_MODEL_WEIGHTS_FILE_ID"
    SCALER_URL: str = "https://drive.google.com/uc?export=download&id=YOUR_SCALER_FILE_ID"
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]  # For development; restrict in production
    
    # Environment (development, production, test)
    ENVIRONMENT: str = "development"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = os.path.join(BASE_DIR, ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()