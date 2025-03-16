from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from streamlit_app.backend.app.routes.prediction import router as prediction_router
from streamlit_app.backend.app.database.connection import connect_to_mongo, close_mongo_connection
from streamlit_app.backend.app.settings.config import settings
from streamlit_app.backend.app.services.ml_model import load_model

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction_router, prefix=settings.API_PREFIX)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialization on application startup"""
    await connect_to_mongo()
    
    # Preload the model to avoid cold start
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not preload model: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    await close_mongo_connection()

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "DON Prediction API is running",
        "status": "healthy",
        "version": settings.API_VERSION
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT
    }