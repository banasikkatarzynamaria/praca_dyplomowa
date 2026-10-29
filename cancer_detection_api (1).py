
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Breast Cancer Detection API",
    description="Leak-free pipeline API with integrated PCA and Random Forest.",
    version="3.0"
)

MODEL_PATH = "best_model.joblib"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class PatientData(BaseModel):
    features: list[float]  # Expects the 30 raw features

@app.get("/")
def read_root():
    return {"message": "Leak-free Breast Cancer Pipeline API is running!"}

@app.post("/predict")
def predict_cancer(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Pipeline model file not found.")

    try:
        # Convert incoming raw features to 2D numpy array
        features_array = np.array(data.features).reshape(1, -1)

        # Pipeline automatically applies PCA preprocessing and predicts
        prediction = model.predict(features_array)

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features_array).tolist()

        return {
            "prediction": int(prediction[0]),
            "probabilities": probability
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
