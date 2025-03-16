import unittest
import sys
import os
from unittest.mock import patch
from pydantic import ValidationError

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.schemas.prediction import PredictionInput, PredictionResponse, PredictionRecord


class TestPredictionSchemas(unittest.TestCase):
    """Tests for prediction schemas"""
    
    def test_prediction_input_valid(self):
        """Test valid prediction input"""
        # Valid input with correct number of features (447)
        valid_data = {"features": [0.1] * 447}
        model = PredictionInput(**valid_data)
        self.assertEqual(len(model.features), 447)
    
    def test_prediction_input_invalid(self):
        """Test invalid prediction input"""
        # Invalid input with wrong number of features
        invalid_data = {"features": [0.1] * 10}  # Too few features
        with self.assertRaises(ValidationError):
            PredictionInput(**invalid_data)
        
        # Invalid input with wrong data type
        invalid_type_data = {"features": ["string"] * 447}  # Strings instead of floats
        with self.assertRaises(ValidationError):
            PredictionInput(**invalid_type_data)
    
    def test_prediction_response_valid(self):
        """Test valid prediction response"""
        valid_data = {
            "id": "507f1f77bcf86cd799439011",
            "don_value": 123.45,
            "status": "success"
        }
        model = PredictionResponse(**valid_data)
        self.assertEqual(model.id, "507f1f77bcf86cd799439011")
        self.assertEqual(model.don_value, 123.45)
        self.assertEqual(model.status, "success")
    
    def test_prediction_record_valid(self):
        """Test valid prediction record"""
        valid_data = {
            "id": "507f1f77bcf86cd799439011",
            "input_data": [0.1] * 447,
            "prediction": 123.45,
            "created_at": "2023-03-16T12:00:00Z"
        }
        model = PredictionRecord(**valid_data)
        self.assertEqual(model.id, "507f1f77bcf86cd799439011")
        self.assertEqual(len(model.input_data), 447)
        self.assertEqual(model.prediction, 123.45)


if __name__ == "__main__":
    unittest.main()