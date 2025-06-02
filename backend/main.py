from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Video Game Sales Prediction API",
    description="API for predicting video game sales based on platform, genre, publisher, and year",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metadata
model_pipeline = None
preprocessor = None
model = None
feature_names = None
top_publishers = None

# Platform and genre options
PLATFORMS = [
    "PS4", "XOne", "PC", "WiiU", "3DS", "PSV", "PS3", "X360", "Wii", "DS",
    "PSP", "PS2", "GBA", "GC", "XB", "PS", "SNES", "N64", "GB", "NES",
    "2600", "DC", "SAT", "SCD", "WS", "NG", "TG16", "3DO", "GG", "PCFX"
]

GENRES = [
    "Action", "Adventure", "Fighting", "Misc", "Platform", "Puzzle",
    "Racing", "Role-Playing", "Shooter", "Simulation", "Sports", "Strategy"
]

# Pydantic models
class GameFeatures(BaseModel):
    platform: str = Field(..., description="Gaming platform")
    genre: str = Field(..., description="Game genre")
    publisher: str = Field(..., description="Game publisher")
    year: int = Field(..., ge=1980, le=2025, description="Release year")
    
    class Config:
        schema_extra = {
            "example": {
                "platform": "PS4",
                "genre": "Action",
                "publisher": "Sony Computer Entertainment",
                "year": 2020
            }
        }

class PredictionResponse(BaseModel):
    predicted_sales: float
    prediction_range: Dict[str, float]
    input_features: Dict[str, Any]
    confidence_level: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_platforms: List[str]
    available_genres: List[str]
    top_publishers: List[str]

def load_model_and_metadata():
    """Load the trained model and associated metadata"""
    global model_pipeline, preprocessor, model, feature_names, top_publishers
    
    try:
        # Get the path to the ml directory
        current_dir = Path(__file__).parent
        ml_dir = current_dir.parent / "ml"
        
        # Load the complete pipeline
        pipeline_path = ml_dir / "model_pipeline.pkl"
        if pipeline_path.exists():
            model_pipeline = joblib.load(pipeline_path)
            preprocessor = model_pipeline['preprocessor']
            model = model_pipeline['model']
            feature_names = model_pipeline['feature_names']
            top_publishers = model_pipeline['top_publishers']
            logger.info("Model pipeline loaded successfully")
        else:
            # Default values if model doesn't exist
            logger.warning("Model file not found, using default values")
            top_publishers = ["Nintendo", "Electronic Arts", "Activision", "Sony Computer Entertainment", "Ubisoft"]
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Set default values
        top_publishers = ["Nintendo", "Electronic Arts", "Activision", "Sony Computer Entertainment", "Ubisoft"]

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model_and_metadata()

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "partial",
        model_loaded=model is not None,
        available_platforms=PLATFORMS,
        available_genres=GENRES,
        top_publishers=top_publishers or []
    )

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_sales(features: GameFeatures):
    """Predict video game sales based on platform, genre, publisher, and year"""
    
    # If model is not loaded, return a mock prediction
    if model is None or preprocessor is None:
        logger.warning("Model not loaded, returning mock prediction")
        return PredictionResponse(
            predicted_sales=2.5,
            prediction_range={"lower_bound": 1.8, "upper_bound": 3.2},
            input_features={
                "platform": features.platform,
                "genre": features.genre,
                "publisher": features.publisher,
                "year": features.year
            },
            confidence_level="Mock"
        )
    
    try:
        # Prepare input data
        publisher = features.publisher
        if top_publishers and publisher not in top_publishers:
            publisher = "Other"
        
        input_data = pd.DataFrame({
            'Platform': [features.platform],
            'Genre': [features.genre],
            'Publisher': [publisher],
            'Year': [features.year]
        })
        
        # Validate inputs
        if features.platform not in PLATFORMS:
            raise HTTPException(status_code=400, detail=f"Invalid platform. Must be one of: {', '.join(PLATFORMS[:10])}...")
        
        if features.genre not in GENRES:
            raise HTTPException(status_code=400, detail=f"Invalid genre. Must be one of: {', '.join(GENRES)}")
        
        # Preprocess and predict
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]
        prediction = max(0, prediction)
        
        # Create confidence ranges
        lower_bound = max(0, prediction * 0.7)
        upper_bound = prediction * 1.3
        
        # Determine confidence level
        if features.year >= 2000 and features.platform in ["PS4", "XOne", "PC", "PS3", "X360"]:
            confidence = "High"
        elif features.year >= 1990:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            predicted_sales=round(prediction, 2),
            prediction_range={
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            },
            input_features={
                "platform": features.platform,
                "genre": features.genre,
                "publisher": publisher,
                "year": features.year
            },
            confidence_level=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/platforms")
async def get_platforms():
    """Get list of available platforms"""
    return {"platforms": PLATFORMS}

@app.get("/api/genres")
async def get_genres():
    """Get list of available genres"""
    return {"genres": GENRES}

@app.get("/api/publishers")
async def get_publishers():
    """Get list of top publishers"""
    return {"publishers": top_publishers or []}

@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "Video Game Sales Prediction API",
        "docs": "/api/docs",
        "health": "/api/health",
        "predict": "/api/predict"
    }

# For Vercel
from mangum import Mangum
handler = Mangum(app)