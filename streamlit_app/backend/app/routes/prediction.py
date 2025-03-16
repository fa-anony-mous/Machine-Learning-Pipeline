from fastapi import APIRouter, HTTPException, Depends
from typing import List

from streamlit_app.backend.app.schemas.prediction import PredictionInput, PredictionResponse, PredictionRecord
from streamlit_app.backend.app.services.ml_model import predict_don
from streamlit_app.backend.app.repositories.prediction import create_prediction, get_prediction_by_id, get_recent_predictions

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=PredictionResponse)
async def create_prediction_route(input_data: PredictionInput):
    """
    Create a new DON prediction
    """
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
    """
    Create a new DON prediction
    """
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/{prediction_id}", response_model=PredictionRecord)
async def get_prediction_route(prediction_id: str):
    """
    Get a prediction by ID
    """
    prediction = await get_prediction_by_id(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail=f"Prediction with ID {prediction_id} not found")
    
    return prediction

@router.get("/", response_model=List[PredictionRecord])
async def list_predictions_route(limit: int = 10):
    """
    Get recent predictions
    """
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
        
    predictions = await get_recent_predictions(limit)
    return predictions