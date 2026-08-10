import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.inference import load_model, predict_image


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="CNN-based binary image classification service",
    version="1.0.0",
)


# ---------------------------------------------------------
# Load model once when the API starts
# ---------------------------------------------------------

model = load_model()


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "status": "running",
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    try:
        contents = await file.read()

        image = Image.open(
            __import__("io").BytesIO(contents)
        )

        result = predict_image(model, image)

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction completed | filename=%s | label=%s | latency_ms=%.2f",
            file.filename,
            result["label"],
            latency_ms,
        )

        return {
            "filename": file.filename,
            "label": result["label"],
            "probabilities": result["probabilities"],
            "latency_ms": round(latency_ms, 2),
        }

    except Exception as exc:
        logger.exception(
            "Prediction failed | filename=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {str(exc)}",
        ) from exc
