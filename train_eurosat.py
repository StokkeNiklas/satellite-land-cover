import os
import numpy as np
import matplotlib.pyplot as plt
from openai import models
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import multiprocessing

# Define paths for already unzipped datasets locally
base_path = "Datasets"
rgb_base = os.path.join(base_path, "EuroSAT_RGB")
ms_base = os.path.join(base_path, "EuroSAT_MS")

# Check if directories exist
print("RGB folder exists:", os.path.exists(rgb_base))
print("MS folder exists:", os.path.exists(ms_base))

# List contents of directories
if os.path.exists(rgb_base):
    print("RGB folder contents:", os.listdir(rgb_base)[:5])
else:
    print("RGB folder not found.")
if os.path.exists(ms_base):
    print("MS folder contents:", os.listdir(ms_base)[:5])
else:
    print("MS folder not found.")

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

# Get sample image paths
rgb_images = find_images(rgb_base) if os.path.exists(rgb_base) else []
ms_images = find_images(ms_base) if os.path.exists(ms_base) else []

print("Sample RGB Images:", rgb_images)
print("Sample MS Images:", ms_images)

# Custom Dataset for Multispectral Images
class MultispectralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print(f"Initializing MultispectralDataset with root: {root_dir}")
        self.dataset = ImageFolder(root_dir, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        with rasterio.open(img_path) as src:
            img = src.read()  # Shape: (13, 64, 64)
            img = np.transpose(img, (1, 2, 0))  # Shape: (64, 64, 13)
            img = img / 10000.0  # Normalize
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Shape: (13, 64, 64)
        if self.transform:
            img = self.transform(img)
        return img, label

# Model for RGB (ResNet18 pretrained)
class RGB_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# Define CNN model for MS (13 channels)
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
        print(f"Data directory {data_dir} not found. Cannot train {model_name}.")
        return None

    print(f"Preparing data for {model_name}...")
    if is_ms:
        dataset = dataset_class(data_dir)
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = ImageFolder(data_dir, transform=transform)

    print(f"Dataset size for {model_name}: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    num_workers = min(8, multiprocessing.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    device = torch.device("cpu")  # CPU-only version
    print(f"Using device: {device}")
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if epoch == 0 or val_acc > max(val_accuracies[:-1]):
            torch.save(model.state_dict(), f"{model_name}.pth")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model.load_state_dict(torch.load(f"{model_name}.pth"))
    return model

if __name__ == "__main__":
    # Place checks here
    print("RGB folder exists:", os.path.exists(rgb_base))
    print("MS folder exists:", os.path.exists(ms_base))

    if os.path.exists(rgb_base):
        print("RGB folder contents:", os.listdir(rgb_base)[:5])
    if os.path.exists(ms_base):
        print("MS folder contents:", os.listdir(ms_base)[:5])

    rgb_images = find_images(rgb_base)
    ms_images = find_images(ms_base)
    print("Sample RGB Images:", rgb_images)
    print("Sample MS Images:", ms_images)

    print("Starting RGB model training...")
    rgb_model = train_and_evaluate(rgb_base, RGBModel, ImageFolder, "rgb_model", is_ms=False)

    print("Starting MS model training...")
    ms_model = train_and_evaluate(ms_base, MSModel, MultispectralDataset, "ms_model", is_ms=True)
