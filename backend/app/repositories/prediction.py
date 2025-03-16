from bson import ObjectId
from datetime import datetime
from typing import List, Optional, Dict, Any

from backend.app.database.connection import get_database

async def create_prediction(input_data: List[float], prediction_result: float) -> str:
    """
    Save prediction data to MongoDB
    
    Args:
        input_data: Input features used for prediction
        prediction_result: Predicted DON value
        
    Returns:
        id: ID of the created document
    """
    db = await get_database()
    
    prediction_doc = {
        "input_data": input_data,
        "prediction": float(prediction_result),
        "created_at": datetime.utcnow()
    }
    
    result = await db.predictions.insert_one(prediction_doc)
    return str(result.inserted_id)

async def get_prediction_by_id(prediction_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a prediction by ID
    
    Args:
        prediction_id: ID of the prediction
        
    Returns:
        prediction: The prediction document or None
    """
    db = await get_database()
    
    try:
        prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
        if prediction:
            prediction["id"] = str(prediction.pop("_id"))
            return prediction
        return None
    except:
        return None

async def get_recent_predictions(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent predictions
    
    Args:
        limit: Maximum number of predictions to return
        
    Returns:
        predictions: List of recent predictions
    """
    db = await get_database()
    
    cursor = db.predictions.find().sort("created_at", -1).limit(limit)
    predictions = []
    
    async for prediction in cursor:
        prediction["id"] = str(prediction.pop("_id"))
        predictions.append(prediction)
        
    return predictions