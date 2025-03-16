from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import importlib.util
import json
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoints for health check
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "DON Prediction API is running on Vercel",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": "production"
    }

# Now let's import the backend functionality carefully
try:
    # First, let's ensure we can import the settings
    from backend.app.settings.config import settings

    # Import key components one by one
    from backend.app.schemas.prediction import PredictionInput, PredictionResponse
    from backend.app.services.ml_model import predict_don, load_model
    from backend.app.repositories.prediction import create_prediction, get_prediction_by_id, get_recent_predictions

    # Flag that imports succeeded
    backend_loaded = True

    # Create prediction endpoint
    @app.post("/api/predictions", response_model=PredictionResponse)
    async def create_prediction_route(input_data: PredictionInput):
        """Create a new DON prediction"""
        try:
            # Validate input length
            if len(input_data.features) != 447:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Expected 447 features, got {len(input_data.features)}"
                )
                
            # Make prediction
            prediction_result = predict_don(input_data.features)
            
            # Save to database
            prediction_id = await create_prediction(input_data.features, prediction_result)
            
            # Return response
            return {
                "id": prediction_id,
                "don_value": prediction_result,
                "status": "success"
            }
        except HTTPException as e:
            # Re-raise HTTPException to return the correct status code
            raise e
        except Exception as e:
            # Handle unexpected errors
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Get prediction by ID
    @app.get("/api/predictions/{prediction_id}")
    async def get_prediction_route(prediction_id: str):
        """Get a prediction by ID"""
        prediction = await get_prediction_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail=f"Prediction with ID {prediction_id} not found")
        
        return prediction

    # List recent predictions
    @app.get("/api/predictions")
    async def list_predictions_route(limit: int = 10):
        """Get recent predictions"""
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
            
        predictions = await get_recent_predictions(limit)
        return predictions

    # Load ML model on startup
    @app.on_event("startup")
    async def startup_event():
        """Initialize the ML model"""
        try:
            load_model()
            print("ML model loaded successfully")
        except Exception as e:
            print(f"Error loading ML model: {e}")

except ImportError as e:
    # Record that backend import failed
    backend_loaded = False
    import_error = str(e)
    
    # Add a diagnostic endpoint to help debug import issues
    @app.get("/api/debug")
    async def debug_info():
        return {
            "backend_loaded": backend_loaded,
            "import_error": import_error,
            "sys_path": sys.path,
            "files_in_backend": os.listdir("backend") if os.path.exists("backend") else "backend dir not found",
            "python_version": sys.version
        }