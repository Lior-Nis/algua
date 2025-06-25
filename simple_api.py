"""
Simple API server for demo purposes.
"""

from datetime import datetime
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Set environment variables
os.environ.setdefault('SECRET_KEY', 'dev-secret-key')
os.environ.setdefault('ALPACA_API_KEY', 'test-api-key')  
os.environ.setdefault('ALPACA_SECRET_KEY', 'test-secret-key')
os.environ.setdefault('ENVIRONMENT', 'development')

from configs.settings import get_settings

app = FastAPI(title="Algua Trading API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping() -> Dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "algua-api"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": 3600,  # Dummy uptime
        "version": "1.0.0",
        "environment": "development",
        "checks": {
            "database": {
                "status": "healthy",
                "message": "Database connection successful",
                "response_time_ms": 15.5,
                "last_check": datetime.utcnow().isoformat()
            },
            "alpaca_api": {
                "status": "healthy", 
                "message": "Alpaca API connection successful",
                "response_time_ms": 45.2,
                "last_check": datetime.utcnow().isoformat()
            },
            "system_resources": {
                "status": "healthy",
                "message": "Resource usage normal: CPU 25%, Memory 45%, Disk 60%",
                "response_time_ms": 1000,
                "last_check": datetime.utcnow().isoformat()
            }
        }
    }

if __name__ == "__main__":
    print("Starting Algua API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)