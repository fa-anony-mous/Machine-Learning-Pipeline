import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directory of the backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Parent directory containing both frontend and backend
PROJECT_ROOT = BASE_DIR.parent

# Path to model artifacts
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
    
    # Model settings
    MODEL_DIR: str = MODEL_DIR
    MODEL_PATH: str = os.path.join(MODEL_DIR, "don_prediction_model.pt")
    SCALER_PATH: str = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    MODEL_INFO_PATH: str = os.path.join(MODEL_DIR, "model_info.json")
    
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