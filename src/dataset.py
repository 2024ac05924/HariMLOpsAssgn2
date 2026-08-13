from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = (224, 224)


class CatsDogsDataset(Dataset):
    """PyTorch dataset for Cats vs Dogs image classification."""

    def __init__(self, csv_file, train=False):
        self.data = pd.read_csv(csv_file)
        self.train = train

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(IMAGE_SIZE),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = Path(row["image_path"].replace("\\", "/"))
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label
