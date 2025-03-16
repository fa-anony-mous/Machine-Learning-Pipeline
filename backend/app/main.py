from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



# Correct import paths
from app.routes.prediction import router as prediction_router
from app.database.connection import connect_to_mongo, close_mongo_connection
from app.settings.config import settings
from app.services.ml_model import load_model

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
    try:
        await connect_to_mongo()
        load_model()  # Preload the model to avoid cold start
    except Exception as e:
        print(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    try:
        await close_mongo_connection()
    except Exception as e:
        print(f"Shutdown error: {e}")

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

# For debugging purposes
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000)