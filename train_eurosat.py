import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

# Define paths for already unzipped datasets locally
base_path = "Datasets"
rgb_base = os.path.join(base_path, "EuroSAT_RGB")
ms_base = os.path.join(base_path, "EuroSAT_MS")

# Function to find images inside class subfolders
def find_images(image_folder, num_samples=5):
    image_paths = []
    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png', '.tif'))]
            image_paths.extend(images[:num_samples])
        if len(image_paths) >= num_samples:
            break
    return image_paths

# Custom Dataset for Multispectral Images
class MultispectralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        with rasterio.open(img_path) as src:
            img = src.read()  # Shape: (13, 64, 64)
            img = np.transpose(img, (1, 2, 0))  # (64, 64, 13)
            img = img / 10000.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (13, 64, 64)
        if self.transform:
            img = self.transform(img)
        return img, label

# Define CNN model for RGB
class RGBModel(nn.Module):
    def __init__(self, num_classes=10):
        super(RGBModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define CNN model for MS
class MSModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MSModel, self).__init__()
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train_and_evaluate(data_dir, model_class, dataset_class, model_name, is_ms=False):
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return None

    if is_ms:
        dataset = dataset_class(data_dir)
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        dataset = ImageFolder(data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cpu")  # Force CPU for compatibility
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}.pth")

    return model

# Execute training for both models
if __name__ == "__main__":
    print("Starting RGB model training...")
    rgb_model = train_and_evaluate(rgb_base, RGBModel, ImageFolder, "rgb_model", is_ms=False)

    print("Starting MS model training...")
    ms_model = train_and_evaluate(ms_base, MSModel, MultispectralDataset, "ms_model", is_ms=True)

    if rgb_model and ms_model:
        print("Both models trained successfully!")
    else:
        print("One or both models failed to train due to missing data or errors.")
