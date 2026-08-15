import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import CatsDogsDataset
from src.model import SimpleCNN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 4
NUM_CLASSES = 2

TRAIN_CSV = "data/processed/train.csv"
VALIDATION_CSV = "data/processed/validation.csv"

MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")

MODEL_PATH = MODEL_DIR / "model.pt"
LOSS_CURVE_PATH = ARTIFACT_DIR / "loss_curve.png"
CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "confusion_matrix.png"


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        all_labels,
        all_predictions,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("CATS VS DOGS - CNN TRAINING")
    print("=" * 60)

    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = CatsDogsDataset(
        TRAIN_CSV,
        train=True,
    )

    validation_dataset = CatsDogsDataset(
        VALIDATION_CSV,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = SimpleCNN(num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -----------------------------------------------------
    # MLflow
    # -----------------------------------------------------

    mlflow.set_experiment("CatsDogsClassification")

    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    all_validation_labels = []
    all_validation_predictions = []

    print("\nStarting MLflow run...")

    with mlflow.start_run() as run:

        mlflow.log_params(
            {
                "model": "SimpleCNN",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "random_state": RANDOM_STATE,
                "image_size": "224x224",
                "optimizer": "Adam",
                "num_classes": NUM_CLASSES,
                "device": str(device),
            }
        )

        # -------------------------------------------------
        # Training loop
        # -------------------------------------------------

        for epoch in range(EPOCHS):

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            (
                validation_loss,
                validation_accuracy,
                validation_labels,
                validation_predictions,
            ) = evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)

            all_validation_labels = validation_labels
            all_validation_predictions = validation_predictions

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
                step=epoch,
            )

            print(
                f"\nEpoch {epoch + 1}/{EPOCHS}"
                f" | Train Loss: {train_loss:.4f}"
                f" | Train Acc: {train_accuracy:.4f}"
                f" | Val Loss: {validation_loss:.4f}"
                f" | Val Acc: {validation_accuracy:.4f}"
            )

        # -------------------------------------------------
        # Save model
        # -------------------------------------------------

        torch.save(model.state_dict(), MODEL_PATH)

        print(f"\nModel saved to: {MODEL_PATH}")

        # -------------------------------------------------
        # Loss curve
        # -------------------------------------------------

        plt.figure(figsize=(8, 5))

        plt.plot(
            range(1, EPOCHS + 1),
            train_losses,
            label="Training Loss",
        )

        plt.plot(
            range(1, EPOCHS + 1),
            validation_losses,
            label="Validation Loss",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()

        plt.savefig(LOSS_CURVE_PATH)
        plt.close()

        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------

        cm = confusion_matrix(
            all_validation_labels,
            all_validation_predictions,
        )

        plt.figure(figsize=(6, 6))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Cat", "Dog"],
        )

        display.plot(values_format="d")
        plt.title("Validation Confusion Matrix")
        plt.tight_layout()

        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.close()

        # -------------------------------------------------
        # MLflow artifacts
        # -------------------------------------------------

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(LOSS_CURVE_PATH))
        mlflow.log_artifact(str(CONFUSION_MATRIX_PATH))

        print(f"MLflow Run ID: {run.info.run_id}")

    print("\nTraining completed successfully.")
    print(f"Final validation accuracy: {validation_accuracies[-1]:.4f}")


if __name__ == "__main__":
    main()
