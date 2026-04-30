import os
import json
import io
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf

MODEL_PATH = os.getenv("MODEL_PATH", "model.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names(foodsg).json")

model = None
class_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    yield


app = FastAPI(title="Food Classification API", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "message": "Food Classification API",
        "model": "MobileNetV3Large trained on FoodSG-233",
        "classes": len(class_names) if class_names else 0,
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]

    results = [
        {"class": class_names[i], "confidence": round(float(predictions[0][i]), 4)}
        for i in top_5_indices
    ]

    return {
        "predicted_class": results[0]["class"],
        "confidence": results[0]["confidence"],
        "top_5": results,
    }
