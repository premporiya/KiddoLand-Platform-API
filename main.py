"""
KiddoLand Backend API
Main application entry point
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import story
from routers import auth
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
