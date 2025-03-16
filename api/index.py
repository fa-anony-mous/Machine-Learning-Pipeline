from fastapi import FastAPI
import sys
import os

# Add the root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your app from the backend
from backend.app.main import app as backend_app

# Create a new app or use the existing one
app = backend_app