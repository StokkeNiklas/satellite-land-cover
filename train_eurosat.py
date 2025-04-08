import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from PIL import Image

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

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

# Split dataset into train, validation, and test sets
def split_dataset(image_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_set, val_set, test_set = [], [], []
    for class_name in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_name)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png', '.tif'))]
            random.shuffle(images)
            train, temp = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
            val, test = train_test_split(temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)
            train_set.extend([(img, class_name) for img in train])
            val_set.extend([(img, class_name) for img in val])
            test_set.extend([(img, class_name) for img in test])
    return train_set, val_set, test_set

# Split both datasets
train_rgb, val_rgb, test_rgb = split_dataset(rgb_base)
train_ms, val_ms, test_ms = split_dataset(ms_base)

# Print dataset sizes
print(f"RGB Dataset Sizes: Train={len(train_rgb)}, Val={len(val_rgb)}, Test={len(test_rgb)}")
print(f"MS Dataset Sizes: Train={len(train_ms)}, Val={len(val_ms)}, Test={len(test_ms)}")

# Custom Dataset class for EuroSAT (RGB or Multispectral)
class EuroSATDataset(Dataset):
    def __init__(self, data, mode="RGB", transform=None):
        self.data = data  # List of (image_path, label)
        self.mode = mode  # "RGB" or "MS"
        self.transform = transform
        self.classes = sorted(set([label for _, label in data]))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        label_idx = self.class_to_idx[label]

        if self.mode == "RGB":
            # Load RGB image using PIL
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        elif self.mode == "MS":
            # Load MS image using rasterio
            with rasterio.open(img_path) as dataset:
                image = dataset.read().astype(np.float32)
                for i in range(image.shape[0]):
                    band_min, band_max = np.min(image[i]), np.max(image[i])
                    if band_max > band_min:
                        image[i] = (image[i] - band_min) / (band_max - band_min)
                image = torch.tensor(image)
            if self.transform:
                image = self.transform(image)

        return image, label_idx

# Define transformations for RGB (ResNet18 requires 224x224 and ImageNet normalization)
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformations for MS
ms_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
])

# Create dataset objects
train_rgb_dataset = EuroSATDataset(train_rgb, mode="RGB", transform=rgb_transform)
val_rgb_dataset = EuroSATDataset(val_rgb, mode="RGB", transform=rgb_transform)
test_rgb_dataset = EuroSATDataset(test_rgb, mode="RGB", transform=rgb_transform)

train_ms_dataset = EuroSATDataset(train_ms, mode="MS", transform=ms_transform)
val_ms_dataset = EuroSATDataset(val_ms, mode="MS", transform=ms_transform)
test_ms_dataset = EuroSATDataset(test_ms, mode="MS", transform=ms_transform)

# Create DataLoaders
BATCH_SIZE = 32

train_rgb_loader = DataLoader(train_rgb_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_rgb_loader = DataLoader(val_rgb_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_rgb_loader = DataLoader(test_rgb_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_ms_loader = DataLoader(train_ms_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_ms_loader = DataLoader(val_ms_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_ms_loader = DataLoader(test_ms_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define ResNet18-based model for RGB
class RGB_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(RGB_Classifier, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define CNN model for MS (13 channels)
class MS_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MS_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)  # Increased dropout for regularization

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function with early stopping
def train_and_evaluate(model, train_loader, val_loader, model_name):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    num_epochs = 15
    patience = 1
    best_val_acc = 0.0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
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

        # Validation
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

        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model and check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} for {model_name}")
                break

    # Plot results
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

    # Load best model
    model.load_state_dict(torch.load(f"{model_name}.pth"))
    return model

# Evaluation function
def evaluate_model(model, test_loader, model_name, classes):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    correct, total = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"{model_name} Test Accuracy: {accuracy:.4f}")

    # Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Instantiate models
    rgb_model = RGB_Classifier(num_classes=10)
    ms_model = MS_Classifier(num_classes=10)

    # Train models
    print("Starting RGB model training...")
    rgb_model = train_and_evaluate(rgb_model, train_rgb_loader, val_rgb_loader, "rgb_model")

    print("Starting MS model training...")
    ms_model = train_and_evaluate(ms_model, train_ms_loader, val_ms_loader, "ms_model")

    # Evaluate models
    classes = sorted(os.listdir(rgb_base))
    print("🔹 Evaluating RGB Model:")
    evaluate_model(rgb_model, test_rgb_loader, "rgb_model", classes)

    print("\n🔹 Evaluating MS Model:")
    evaluate_model(ms_model, test_ms_loader, "ms_model", classes)

    print("Models trained and evaluated successfully!")