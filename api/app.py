import io
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

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
# Prometheus metrics
# ---------------------------------------------------------

REQUEST_COUNT = Counter(
    "cats_dogs_requests_total",
    "Total number of prediction requests",
)

SUCCESS_COUNT = Counter(
    "cats_dogs_predictions_total",
    "Total number of successful predictions",
)

ERROR_COUNT = Counter(
    "cats_dogs_errors_total",
    "Total number of failed prediction requests",
)

REQUEST_LATENCY = Histogram(
    "cats_dogs_request_latency_seconds",
    "Prediction request latency in seconds",
)


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="CNN-based binary image classification service",
    version="1.0.0",
)


# ---------------------------------------------------------
# Load model once when API starts
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
# Prometheus metrics endpoint
# ---------------------------------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()

    start_time = time.perf_counter()

    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents))

        result = predict_image(model, image)

        latency_seconds = time.perf_counter() - start_time
        latency_ms = latency_seconds * 1000

        SUCCESS_COUNT.inc()
        REQUEST_LATENCY.observe(latency_seconds)

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
        ERROR_COUNT.inc()

        latency_seconds = time.perf_counter() - start_time
        REQUEST_LATENCY.observe(latency_seconds)

        logger.exception(
            "Prediction failed | filename=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {str(exc)}",
        ) from exc