# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# dataset = datasets.ImageFolder(
#     root="data",
#     transform=transform
# )

# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# images, labels = next(iter(loader))
# print(images.shape)  # [32, 3, 224, 224]
# print(labels)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_model
import os

# 1. SETUP DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. DATA PREPARATION
# We define the transforms (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DEFINING PATHS
# Make sure your folders are named exactly 'train' and 'test' inside the 'data' folder
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

print(f"Loading training data from: {TRAIN_DIR}")
print(f"Loading testing data from: {TEST_DIR}")

# Load the datasets directly from the folders
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

# Create the loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print class mapping to confirm (0=AI, 1=Real or vice versa)
print(f"Class Mapping: {train_dataset.class_to_idx}")

# 3. SETUP MODEL
model = get_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 4. TRAINING LOOP
epochs = 5
print("\nStarting Training...")

for epoch in range(epochs):
    model.train() # Set to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")

    # Validation (Optional: Check accuracy on test set after each epoch)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 5. SAVE
# Save inside the 'src' folder or main folder
torch.save(model.state_dict(), "model.pt")
print("\nModel saved as model.pt")