from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from typing import List, Dict, Any
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Now import the backend functionality with better error handling
try:
    # Make sure backend module structure exists
    if not os.path.exists("backend"):
        logger.error("Backend directory not found")
        raise ImportError("Backend directory not found")
    
    # Create utils directory if it doesn't exist
    os.makedirs("backend/app/utils", exist_ok=True)
    
    # First, ensure we can import settings
    from backend.app.settings.config import settings
    logger.info("Successfully imported settings")
    
    # Create proper imports
    try:
        from backend.app.schemas.prediction import PredictionInput, PredictionResponse
        logger.info("Successfully imported schemas")
    except ImportError as e:
        logger.error(f"Error importing schemas: {e}")
        raise
    
    try:
        from backend.app.services.ml_model import predict_don, load_model
        logger.info("Successfully imported ML model")
    except ImportError as e:
        logger.error(f"Error importing ML model: {e}")
        raise
        
    try:
        from backend.app.repositories.prediction import create_prediction, get_prediction_by_id, get_recent_predictions
        logger.info("Successfully imported repositories")
    except ImportError as e:
        logger.error(f"Error importing repositories: {e}")
        raise
        
    try:
        from backend.app.database.connection import connect_to_mongo, close_mongo_connection
        logger.info("Successfully imported database connection")
    except ImportError as e:
        logger.error(f"Error importing database connection: {e}")
        raise

    # Flag that imports succeeded
    backend_loaded = True
    logger.info("Backend successfully loaded")

    # Create prediction endpoint
    @app.post("/api/predictions", response_model=PredictionResponse)
    async def create_prediction_route(input_data: PredictionInput):
        """Create a new DON prediction"""
        try:
            logger.info(f"Received prediction request with {len(input_data.features)} features")
            # Validate input length
            if len(input_data.features) != 447:
                logger.warning(f"Invalid feature count: {len(input_data.features)}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Expected 447 features, got {len(input_data.features)}"
                )
                
            # Make prediction
            logger.info("Making prediction...")
            prediction_result = predict_don(input_data.features)
            logger.info(f"Prediction result: {prediction_result}")
            
            # Save to database
            logger.info("Saving prediction to database...")
            prediction_id = await create_prediction(input_data.features, prediction_result)
            logger.info(f"Prediction saved with ID: {prediction_id}")
            
            # Return response
            return {
                "id": prediction_id,
                "don_value": prediction_result,
                "status": "success"
            }
        except HTTPException as e:
            # Re-raise HTTPException to return the correct status code
            logger.warning(f"HTTP Exception: {e.detail}")
            raise e
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Get prediction by ID
    @app.get("/api/predictions/{prediction_id}")
    async def get_prediction_route(prediction_id: str):
        """Get a prediction by ID"""
        logger.info(f"Getting prediction with ID: {prediction_id}")
        prediction = await get_prediction_by_id(prediction_id)
        if not prediction:
            logger.warning(f"Prediction not found: {prediction_id}")
            raise HTTPException(status_code=404, detail=f"Prediction with ID {prediction_id} not found")
        
        logger.info(f"Retrieved prediction: {prediction}")
        return prediction

    # List recent predictions
    @app.get("/api/predictions")
    async def list_predictions_route(limit: int = 10):
        """Get recent predictions"""
        if limit < 1 or limit > 100:
            logger.warning(f"Invalid limit: {limit}")
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
            
        logger.info(f"Getting recent predictions (limit: {limit})")
        predictions = await get_recent_predictions(limit)
        logger.info(f"Retrieved {len(predictions)} predictions")
        return predictions

    # Connect to MongoDB on startup
    @app.on_event("startup")
    async def startup_event():
        """Initialize the database connection and optionally preload ML model"""
        try:
            logger.info("Connecting to MongoDB...")
            await connect_to_mongo()
            logger.info("MongoDB connection established")
            
            # Uncomment to preload the ML model (optional)
            # logger.info("Preloading ML model...")
            # load_model()
            # logger.info("ML model loaded successfully")
        except Exception as e:
            logger.error(f"Error during startup: {e}", exc_info=True)
            print(f"Error during startup: {e}")

    # Disconnect from MongoDB on shutdown
    @app.on_event("shutdown")
    async def shutdown_event():
        """Close the database connection"""
        try:
            logger.info("Closing MongoDB connection...")
            await close_mongo_connection()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

except ImportError as e:
    # Record that backend import failed
    backend_loaded = False
    import_error = str(e)
    logger.error(f"Backend import failed: {import_error}")
    
    # Add a diagnostic endpoint to help debug import issues
    @app.get("/api/debug")
    async def debug_info():
        try:
            backend_files = os.listdir("backend/app") if os.path.exists("backend/app") else "backend/app dir not found"
        except Exception as e:
            backend_files = f"Error listing files: {str(e)}"
            
        return {
            "backend_loaded": backend_loaded,
            "import_error": import_error,
            "sys_path": sys.path,
            "files_in_backend": backend_files,
            "python_version": sys.version,
            "environment": os.environ.get("ENVIRONMENT", "Not set")
        }