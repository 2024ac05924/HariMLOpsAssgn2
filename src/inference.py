from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import SimpleCNN


IMAGE_SIZE = (224, 224)

CLASS_NAMES = {
    0: "Cat",
    1: "Dog",
}

MODEL_PATH = Path("models/model.pt")


def get_transform():
    """Return the inference preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(model_path=MODEL_PATH):
    """Load the trained CNN model."""
    model = SimpleCNN(num_classes=2)

    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_image(model, image):
    """Predict Cat/Dog class and return probabilities."""
    image = image.convert("RGB")

    transform = get_transform()

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    predicted_index = int(torch.argmax(probabilities, dim=1).item())

    probability_values = probabilities[0].tolist()

    return {
        "label": CLASS_NAMES[predicted_index],
        "probabilities": {
            "Cat": float(probability_values[0]),
            "Dog": float(probability_values[1]),
        },
    }
