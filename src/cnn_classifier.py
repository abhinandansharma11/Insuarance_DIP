import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
from typing import List, Tuple

class DocumentDataset(Dataset):
    """Custom PyTorch Dataset loader for synthetic insurance documents."""
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DocumentCNN(nn.Module):
    """Lightweight Convolutional Neural Network for Document Classification."""
    def __init__(self, num_classes: int = 3):
        super(DocumentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 125 * 125, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DocumentClassifierTrainer:
    """Manages training and evaluation for both PyTorch CNN and HOG+SVM baselines."""

    def train_cnn_model(self, dataloader: DataLoader, num_epochs: int = 5, lr: float = 0.001) -> nn.Module:
        """Trains the Document CNN model."""
        model = DocumentCNN(num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

        return model

    def extract_hog_features(self, image_paths: List[str]) -> np.ndarray:
        """Extracts Histogram of Oriented Gradients (HOG) features for traditional ML baseline."""
        features = []
        for path in image_paths:
            img = Image.open(path).convert("L").resize((128, 128))
            np_img = np.array(img)
            fd = hog(np_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
            features.append(fd)
        return np.array(features)

    def train_svm_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """Trains a Support Vector Machine classifier on HOG features."""
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)
        return clf