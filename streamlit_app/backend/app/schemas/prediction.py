from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional
from datetime import datetime

class PredictionInput(BaseModel):
    """Input schema for DON prediction"""
    features: List[float] = Field(..., min_items=447, max_items=447)

    @field_validator('features')
    def validate_features(cls, v):
        """Validate that features list has exactly 447 items"""
        if len(v) != 447:
            raise ValueError("Expected 447 features")
        return v

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    id: str = Field(..., description="Unique identifier for the prediction")
    don_value: float = Field(..., description="Predicted DON value")
    status: str = Field(..., description="Status of the prediction")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "5f8a1d5e9b7c6a3d2e1f0a9b",
                "don_value": 1234.56,
                "status": "success",
                "timestamp": "2023-03-16T12:00:00Z"
            }
        }

class PredictionRecord(BaseModel):
    """Schema for a stored prediction record"""
    id: str = Field(..., description="Unique identifier for the prediction")
    input_data: List[float] = Field(..., description="Input features used for prediction")
    prediction: float = Field(..., description="Predicted DON value")
    created_at: datetime = Field(..., description="When the prediction was made")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "5f8a1d5e9b7c6a3d2e1f0a9b",
                "input_data": [0.1, -0.2, 0.3, 0.4, 0.5] + [0.0] * 442,
                "prediction": 1234.56,
                "created_at": "2023-03-16T12:00:00Z"
            }
        }