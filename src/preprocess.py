import random
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# Paths
RAW_DIR = Path("data/raw/PetImages")
PROCESSED_DIR = Path("data/processed")

# Reproducibility
RANDOM_STATE = 42

# Image settings required by the assignment
IMAGE_SIZE = (224, 224)
IMAGE_MODE = "RGB"


def validate_image(image_path):
    """Check whether an image can be opened and converted to RGB."""
    try:
        with Image.open(image_path) as image:
            image.convert(IMAGE_MODE)
        return True
    except Exception:
        return False


def collect_images():
    """Collect valid Cat and Dog image paths with binary labels."""
    records = []
    invalid_files = []

    class_mapping = {
        "Cat": 0,
        "Dog": 1,
    }

    for class_name, label in class_mapping.items():
        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing directory: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue

            if validate_image(image_path):
                records.append(
                    {
                        "image_path": str(image_path),
                        "label": label,
                        "class_name": class_name,
                    }
                )
            else:
                invalid_files.append(str(image_path))

    print(f"Valid images: {len(records)}")
    print(f"Invalid images: {len(invalid_files)}")

    if invalid_files:
        print("\nInvalid files:")
        for path in invalid_files[:20]:
            print(path)

        if len(invalid_files) > 20:
            print(f"... and {len(invalid_files) - 20} more")

    return pd.DataFrame(records)


def create_splits(df):
    """Create reproducible 80/10/10 train/validation/test splits."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    return train_df, validation_df, test_df


def save_splits(train_df, validation_df, test_df):
    """Save split manifests."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    validation_df = validation_df.sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)


def print_summary(train_df, validation_df, test_df):
    """Print split sizes and class distributions."""
    print("\nDataset summary")
    print("-" * 50)

    for name, split_df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        print(f"\n{name}: {len(split_df)} images")
        print(split_df["class_name"].value_counts().sort_index())

    total = len(train_df) + len(validation_df) + len(test_df)

    print("\nTotal images:", total)
    print("Train percentage:", round(len(train_df) / total * 100, 2), "%")
    print("Validation percentage:", round(len(validation_df) / total * 100, 2), "%")
    print("Test percentage:", round(len(test_df) / total * 100, 2), "%")
    print(f"\nTarget image size: {IMAGE_SIZE}")
    print(f"Target image mode: {IMAGE_MODE}")


def main():
    print("=" * 60)
    print("CATS VS DOGS - DATA PREPROCESSING")
    print("=" * 60)

    print("\nCollecting and validating images...")
    df = collect_images()

    if df.empty:
        raise RuntimeError("No valid images were found.")

    print("\nCreating 80/10/10 train/validation/test split...")
    train_df, validation_df, test_df = create_splits(df)

    print("\nSaving split manifests...")
    save_splits(train_df, validation_df, test_df)

    print_summary(train_df, validation_df, test_df)

    print("\nPreprocessing completed successfully.")
    print(f"Processed metadata saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
