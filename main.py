"""
KiddoLand Backend API
Main application entry point
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import story
from routers import auth
from routers import ai
from dotenv import load_dotenv
from utils.config import validate_huggingface_config

# Load environment variables
load_dotenv()

# Fail fast if Hugging Face configuration is missing
validate_huggingface_config()

app = FastAPI(
    title="KiddoLand API",
    description="AI-powered story generation and rewriting for children aged 1-18",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(story.router, prefix="/story", tags=["Story"])
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(ai.router, prefix="/ai", tags=["AI"])

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "KiddoLand API",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=8000)
