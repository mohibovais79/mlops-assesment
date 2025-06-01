import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import asynccontextmanager
from PIL import Image

from inference.predictor import ImagePreprocessor, ONNXModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model and preprocessor when the application starts.
    """

    global preprocessor, onnx_model, ONNX_MODEL_PATH  

    print("Application startup: Initializing model and preprocessor...")
    try:
        preprocessor = ImagePreprocessor()

        if not os.path.exists(ONNX_MODEL_PATH):
            error_msg = f"Critical Error: ONNX model file not found at '{ONNX_MODEL_PATH}' during startup."
            print(error_msg)
            raise RuntimeError(error_msg)

        onnx_model = ONNXModel(model_path=ONNX_MODEL_PATH)
        print("Model and preprocessor initialized successfully.")
    except Exception as e:
        print(f"Critical Error during startup model initialization: {e}")
        preprocessor = None
        onnx_model = None
        raise
    yield

    print("Application shutdown: Cleaning up resources (if any)...")
    preprocessor = None
    onnx_model = None


app = FastAPI(title="Image Classification App", lifespan=lifespan)


ONNX_MODEL_PATH = "weights/image_classifier.onnx"
preprocessor = None
onnx_model = None


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file, preprocesses it, runs inference,
    and returns the predicted class ID and probabilities.
    """
    global preprocessor, onnx_model

    if preprocessor is None or onnx_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server not ready.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    if pil_image is None:
        raise HTTPException(status_code=400, detail="Could not read image from uploaded file.")

    try:
        preprocessed_data = preprocessor.preprocess_pil_image(pil_image)

        predictions = onnx_model.predict(preprocessed_data)

        probabilities = predictions[0].tolist()
        predicted_class_id = int(np.argmax(predictions[0]))

        return {
            "predicted_class_id": predicted_class_id,
            "probabilities": probabilities,  #  list of 1000 probabilities
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    if preprocessor is not None and onnx_model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False, "detail": "Model or preprocessor not initialized."}
