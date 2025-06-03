from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
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
    docs_url="/",  # Swagger UI at root
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
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

# Get available options for dropdowns
PLATFORMS = [
    "PS4", "XOne", "PC", "WiiU", "3DS", "PSV", "PS3", "X360", "Wii", "DS",
    "PSP", "PS2", "GBA", "GC", "XB", "PS", "SNES", "N64", "GB", "NES",
    "2600", "DC", "SAT", "SCD", "WS", "NG", "TG16", "3DO", "GG", "PCFX"
]

GENRES = [
    "Action", "Adventure", "Fighting", "Misc", "Platform", "Puzzle",
    "Racing", "Role-Playing", "Shooter", "Simulation", "Sports", "Strategy"
]

# Pydantic models for request/response
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

class ModelInfoResponse(BaseModel):
    model_type: str
    feature_count: int
    training_data_size: str
    available_platforms: List[str]
    available_genres: List[str]
    sample_predictions: List[Dict]

def load_model_and_metadata():
    """Load the trained model and associated metadata"""
    global model_pipeline, preprocessor, model, feature_names, top_publishers
    
    try:
        # Get the path to the ml directory
        ml_dir = Path(__file__).parent.parent / "ml"
        
        # Load the complete pipeline
        pipeline_path = ml_dir / "model_pipeline.pkl"
        if pipeline_path.exists():
            model_pipeline = joblib.load(pipeline_path)
            preprocessor = model_pipeline['preprocessor']
            model = model_pipeline['model']
            feature_names = model_pipeline['feature_names']
            top_publishers = model_pipeline['top_publishers']
        else:
            # Load individual components if pipeline doesn't exist
            model_path = ml_dir / "model.pkl"
            preprocessor_path = ml_dir / "preprocessor.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            model = joblib.load(model_path)
            
            if preprocessor_path.exists():
                preprocessor = joblib.load(preprocessor_path)
                
            # Load feature names
            feature_names_path = ml_dir / "feature_names.pkl"
            if feature_names_path.exists():
                feature_names = joblib.load(feature_names_path)
                
            # Load top publishers
            top_publishers_path = ml_dir / "top_publishers.pkl"
            if top_publishers_path.exists():
                top_publishers = joblib.load(top_publishers_path)
            else:
                top_publishers = ["Nintendo", "Electronic Arts", "Activision", "Sony Computer Entertainment", "Ubisoft"]
        
        logger.info("Model and preprocessor loaded successfully")
        logger.info(f"Feature names count: {len(feature_names) if feature_names else 'Not loaded'}")
        logger.info(f"Top publishers count: {len(top_publishers) if top_publishers else 'Not loaded'}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model_and_metadata()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        available_platforms=PLATFORMS,
        available_genres=GENRES,
        top_publishers=top_publishers or []
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Load sample predictions if available
    sample_predictions = []
    try:
        ml_dir = Path(__file__).parent.parent / "ml"
        sample_path = ml_dir / "sample_predictions.csv"
        if sample_path.exists():
            import pandas as pd
            sample_df = pd.read_csv(sample_path)
            sample_predictions = sample_df.head(3).to_dict('records')
    except Exception as e:
        logger.warning(f"Could not load sample predictions: {e}")
    
    return ModelInfoResponse(
        model_type="Random Forest Regressor",
        feature_count=len(feature_names) if feature_names else 0,
        training_data_size="~16,000 video games",
        available_platforms=PLATFORMS,
        available_genres=GENRES,
        sample_predictions=sample_predictions
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(features: GameFeatures):
    """
    Predict video game sales based on platform, genre, publisher, and year
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")
    
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
            raise HTTPException(status_code=400, detail=f"Invalid platform. Must be one of: {', '.join(PLATFORMS)}")
        
        if features.genre not in GENRES:
            raise HTTPException(status_code=400, detail=f"Invalid genre. Must be one of: {', '.join(GENRES)}")
        
        # Preprocess the data
        input_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        
        # Ensure prediction is non-negative
        prediction = max(0, prediction)
        
        # Create confidence ranges (rough estimation based on typical model uncertainty)
        lower_bound = max(0, prediction * 0.7)
        upper_bound = prediction * 1.3
        
        # Determine confidence level
        if features.year >= 2000 and features.platform in ["PS4", "XOne", "PC", "PS3", "X360"]:
            confidence = "High"
        elif features.year >= 1990:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        logger.info(f"Prediction made: {prediction:.2f}M for {features.platform} {features.genre} game by {publisher} in {features.year}")
        
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

@app.get("/platforms")
async def get_platforms():
    """Get list of available platforms"""
    return {"platforms": PLATFORMS}

@app.get("/genres")
async def get_genres():
    """Get list of available genres"""
    return {"genres": GENRES}

@app.get("/publishers")
async def get_publishers():
    """Get list of top publishers"""
    return {"publishers": top_publishers or []}

@app.get("/")
async def root():
    """Root endpoint - redirects to docs"""
    return {
        "message": "Video Game Sales Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)