from pathlib import Path

from PIL import Image

from src.dataset import CatsDogsDataset
from src.preprocess import validate_image


def test_validate_image(tmp_path):
    image_path = Path(tmp_path) / "test_image.jpg"

    image = Image.new("RGB", (100, 100), color="white")
    image.save(image_path)

    assert validate_image(image_path) is True


def test_training_dataset_returns_224_rgb_tensor():
    dataset = CatsDogsDataset(
        "data/processed/train.csv",
        train=True,
    )

    image, label = dataset[0]

    assert tuple(image.shape) == (3, 224, 224)
    assert label in (0, 1)
