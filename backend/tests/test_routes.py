import unittest
import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.schemas.prediction import PredictionInput

client = TestClient(app)


class TestPredictionRoutes(unittest.TestCase):
    """Tests for the prediction API routes"""

    def setUp(self):
        """Setup test data"""
        self.test_features = [0.1] * 447
        self.test_prediction = 120.5
        self.test_id = "507f1f77bcf86cd799439011"
        self.prediction_data = {
            "features": self.test_features
        }
        
        # Sample prediction return data
        self.prediction_response = {
            "id": self.test_id,
            "don_value": self.test_prediction,
            "status": "success",
            "timestamp": "2023-03-16T12:00:00Z"
        }
        
        # Mock for repository returns
        self.mock_prediction = {
            "id": self.test_id,
            "input_data": self.test_features,
            "prediction": self.test_prediction,
            "created_at": "2023-03-16T12:00:00Z"
        }

    @patch('app.routes.prediction.predict_don')
    @patch('app.routes.prediction.create_prediction')
    def test_create_prediction_route(self, mock_create, mock_predict):
        """Test creating a new prediction via API"""
        # Setup mocks
        mock_predict.return_value = self.test_prediction
        mock_create.return_value = self.test_id  # Return actual ID instead of AsyncMock
        
        # Make the request
        response = client.post("/predictions/", json=self.prediction_data)
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["don_value"], self.test_prediction)
        self.assertEqual(response.json()["id"], self.test_id)
        mock_predict.assert_called_once_with(self.test_features)
    
        def test_create_prediction_invalid_features(self):
            """Test validation error when features count is wrong"""
            # Make request with wrong number of features
            response = client.post("/predictions/", json={"features": [0.1] * 10})
            
            # Assertions
            self.assertEqual(response.status_code, 422)  # Expect 422 instead of 400
            self.assertIn("Expected 447 features", response.json()["detail"][0]["msg"])

    @patch('app.routes.prediction.get_prediction_by_id')
    def test_get_prediction_route(self, mock_get):
        """Test getting a prediction by ID"""
        # Setup mock
        mock_get.return_value = self.mock_prediction  # Return actual dict instead of AsyncMock
        
        # Make the request
        response = client.get(f"/predictions/{self.test_id}")
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], self.test_id)
        self.assertEqual(response.json()["prediction"], self.test_prediction)
        mock_get.assert_called_once_with(self.test_id)
    
    @patch('app.routes.prediction.get_prediction_by_id')
    def test_get_prediction_not_found(self, mock_get):
        """Test getting a non-existent prediction"""
        # Setup mock
        mock_get.return_value = None  # Return None instead of AsyncMock
        
        # Make the request
        response = client.get(f"/predictions/{self.test_id}")
        
        # Assertions
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])
        mock_get.assert_called_once_with(self.test_id)

    @patch('app.routes.prediction.get_recent_predictions')
    def test_list_predictions_route(self, mock_list):
        """Test listing recent predictions"""
        # Setup mock
        mock_list.return_value = [self.mock_prediction]  # Return actual list instead of AsyncMock
        
        # Make the request
        response = client.get("/predictions/")
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["id"], self.test_id)
        mock_list.assert_called_once_with(10)  # Default limit is 10
    
    @patch('app.routes.prediction.get_recent_predictions')
    def test_list_predictions_with_limit(self, mock_list):
        """Test listing predictions with custom limit"""
        # Setup mock
        mock_list.return_value = [self.mock_prediction] * 5  # Return actual list instead of AsyncMock
        
        # Make the request
        response = client.get("/predictions/?limit=5")
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 5)
        mock_list.assert_called_once_with(5)

    def test_list_predictions_invalid_limit(self):
        """Test validation error when limit is invalid"""
        # Make request with invalid limit
        response = client.get("/predictions/?limit=0")
        
        # Assertions
        self.assertEqual(response.status_code, 400)
        self.assertIn("Limit must be between", response.json()["detail"])
        
        # Make request with too large limit
        response = client.get("/predictions/?limit=101")
        
        # Assertions
        self.assertEqual(response.status_code, 400)
        self.assertIn("Limit must be between", response.json()["detail"])


if __name__ == '__main__':
    unittest.main()