import motor.motor_asyncio
from fastapi import Depends

from backend.app.settings.config import settings

# Database client and connection
client = None
db = None

async def connect_to_mongo():
    """Connect to MongoDB database"""
    global client, db
    
    if client is None:
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGO_URI)
        db = client[settings.DB_NAME]
        print(f"Connected to MongoDB at {settings.MONGO_URI}")

async def close_mongo_connection():
    """Close MongoDB connection"""
    global client
    
    if client:
        client.close()
        print("MongoDB connection closed")

async def get_database():
    """Get database instance for dependency injection"""
    global db
    
    if db is None:
        await connect_to_mongo()
    
    return db