import sys
import os

# Add the backend directory to the path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your FastAPI app
from backend.app.main import app

# This is what Vercel will use
handler = app