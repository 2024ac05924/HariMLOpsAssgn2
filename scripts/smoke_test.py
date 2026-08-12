import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("SMOKE_TEST_URL", "http://localhost:8002")
IMAGE_PATH = Path("data/raw/PetImages/Cat/1.jpg")


def check_health():
    print("Checking health endpoint...")

    response = requests.get(f"{BASE_URL}/health", timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "healthy":
        raise AssertionError(f"Unexpected health status: {data}")

    if data.get("model_loaded") is not True:
        raise AssertionError(f"Model is not loaded: {data}")

    print("Health check: PASSED")


def check_prediction():
    print("Checking prediction endpoint...")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

    start_time = time.perf_counter()

    with IMAGE_PATH.open("rb") as image_file:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("1.jpg", image_file, "image/jpeg")},
            timeout=30,
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    response.raise_for_status()

    data = response.json()

    if data.get("label") not in {"Cat", "Dog"}:
        raise AssertionError(f"Invalid prediction label: {data}")

    probabilities = data.get("probabilities", {})

    if "Cat" not in probabilities or "Dog" not in probabilities:
        raise AssertionError(f"Missing class probabilities: {data}")

    print(f"Prediction: {data['label']}")
    print(f"Probabilities: {probabilities}")
    print(f"Smoke test prediction latency: {elapsed_ms:.2f} ms")
    print("Prediction check: PASSED")


def main():
    try:
        check_health()
        check_prediction()
        print("\nSmoke tests PASSED")
        return 0

    except Exception as exc:
        print(f"\nSmoke tests FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())