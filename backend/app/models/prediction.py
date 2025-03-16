class PredictionModel:
    def __init__(self, id=None, input_data=None, prediction=None, created_at=None):
        self.id = id
        self.input_data = input_data
        self.prediction = prediction
        self.created_at = created_at
    
    @classmethod
    def from_db(cls, data):
        """Create model from database document"""
        return cls(
            id=str(data.get("_id")),
            input_data=data.get("input_data"),
            prediction=data.get("prediction"),
            created_at=data.get("created_at")
        )
    
    def to_db(self):
        """Convert to database document"""
        return {
            "input_data": self.input_data,
            "prediction": self.prediction,
            "created_at": self.created_at
        }