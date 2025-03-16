import unittest
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from bson import ObjectId

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.repositories.prediction import create_prediction, get_prediction_by_id, get_recent_predictions


class TestPredictionRepository(unittest.TestCase):
    """Tests for the prediction repository"""

    def setUp(self):
        """Setup test data"""
        self.test_features = [0.1] * 447
        self.test_prediction = 120.5
        self.test_id = "507f1f77bcf86cd799439011"
        self.test_object_id = ObjectId(self.test_id)
        
        # Sample prediction document
        self.prediction_doc = {
            "_id": self.test_object_id,
            "input_data": self.test_features,
            "prediction": self.test_prediction,
            "created_at": datetime.utcnow()
        }
        
        # Expected document after transformation
        self.expected_doc = {
            "id": self.test_id,
            "input_data": self.test_features,
            "prediction": self.test_prediction,
            "created_at": self.prediction_doc["created_at"]
        }

    @pytest.mark.asyncio
    @patch('app.repositories.prediction.get_database')
    async def test_create_prediction(self, mock_get_db):
        """Test creating a prediction"""
        # Setup mock
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id=self.test_object_id)
        mock_db.predictions = mock_collection
        mock_get_db.return_value = mock_db
        
        # Call the function
        result = await create_prediction(self.test_features, self.test_prediction)
        
        # Assertions
        self.assertEqual(result, self.test_id)
        mock_collection.insert_one.assert_called_once()
        
        # Check that the document was created with the right structure
        call_args = mock_collection.insert_one.call_args[0][0]
        self.assertEqual(call_args["input_data"], self.test_features)
        self.assertEqual(call_args["prediction"], self.test_prediction)
        self.assertIn("created_at", call_args)

    @pytest.mark.asyncio
    @patch('app.repositories.prediction.get_database')
    async def test_get_prediction_by_id(self, mock_get_db):
        """Test retrieving a prediction by ID"""
        # Setup mock
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = self.prediction_doc
        mock_db.predictions = mock_collection
        mock_get_db.return_value = mock_db
        
        # Call the function
        result = await get_prediction_by_id(self.test_id)
        
        # Assertions
        self.assertEqual(result["id"], self.test_id)
        self.assertEqual(result["prediction"], self.test_prediction)
        self.assertEqual(result["input_data"], self.test_features)
        mock_collection.find_one.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.repositories.prediction.get_database')
    async def test_get_prediction_not_found(self, mock_get_db):
        """Test retrieving a non-existent prediction"""
        # Setup mock
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_db.predictions = mock_collection
        mock_get_db.return_value = mock_db
        
        # Call the function
        result = await get_prediction_by_id(self.test_id)
        
        # Assertions
        self.assertIsNone(result)
        mock_collection.find_one.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.repositories.prediction.get_database')
    async def test_get_recent_predictions(self, mock_get_db):
        """Test retrieving recent predictions"""
        # Setup mock
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        
        # Create multiple prediction documents
        predictions = [
            {
                "_id": ObjectId(),
                "input_data": self.test_features,
                "prediction": self.test_prediction + i,
                "created_at": datetime.utcnow()
            }
            for i in range(3)
        ]
        
        # Make the cursor work like an async iterator
        mock_cursor.__aiter__.return_value = predictions
        
        # Mock the find method to return our cursor
        mock_collection.find.return_value = mock_cursor
        mock_db.predictions = mock_collection
        mock_get_db.return_value = mock_db
        
        # Call the function
        limit = 10
        results = await get_recent_predictions(limit)
        
        # Assertions
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result["prediction"], self.test_prediction + i)
            self.assertEqual(result["input_data"], self.test_features)
            self.assertIn("id", result)
            self.assertIn("created_at", result)
        
        # Check that find was called with the right arguments
        mock_collection.find.assert_called_once()
        mock_cursor.sort.assert_called_once_with("created_at", -1)
        mock_cursor.limit.assert_called_once_with(limit)


if __name__ == '__main__':
    unittest.main()