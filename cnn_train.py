import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

print("🔄 Initializing Full Machine Learning Training Pipeline...")

# 1. Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

TRAIN_DIR = "train"
VAL_DIR = "val"

train_dataset = ImageFolder(root=TRAIN_DIR, transform=data_transforms['train'])
val_dataset = ImageFolder(root=VAL_DIR, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"✅ Loaded {num_classes} Tomato Disease Classes successfully!")

# 2. Build Model
model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)

# Freeze lower layers to speed up training on laptop CPU
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 3. Full Training Loop
EPOCHS = 5
print(f"\n🚀 Starting Full Training for {EPOCHS} Epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress update every 20 batches
        if (i + 1) % 20 == 0:
            print(f"Batch [{i+1}/{len(train_loader)}] | Loss: {running_loss / (i+1):.4f} | Train Acc: {100. * correct / total:.2f}%")
            
    # Validation phase after each epoch
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
    print(f"🌟 Epoch {epoch+1} Finished | Final Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {100. * val_correct / val_total:.2f}%")

# 4. Save final optimized model
torch.save(model.state_dict(), "tomato_model_final.pth")
print("\n🎉 Training Complete! Optimized model saved as 'tomato_model_final.pth'!")