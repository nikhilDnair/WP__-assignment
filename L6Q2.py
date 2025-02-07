import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define image transformations for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure images are resized to 224x224 (fixed size)
    transforms.RandomHorizontalFlip(),  # Data augmentation (flipping images)
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for AlexNet
])

validation_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure images are resized to 224x224 (fixed size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for AlexNet
])

# Load the datasets
train_dir = './cats_and_dogs_filtered/train'
val_dir = './cats_and_dogs_filtered/validation'

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=validation_transforms)

# Load data into DataLoader
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Load the pre-trained AlexNet model
alexnet = models.alexnet(weights="IMAGENET1K_V1")  # Using the pretrained weights

# Freeze all layers so no weights are updated except for the final classifier
for param in alexnet.parameters():
    param.requires_grad = False

# Modify the final layer (classifier) to have two outputs (for cats and dogs)
alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.classifier[6].parameters(), lr=0.001)  # Only train the last layer

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    alexnet.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Evaluate the model
alexnet.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = alexnet(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

validation_accuracy = 100 * correct / total
print(f"Validation Accuracy: {validation_accuracy:.2f}%")

# Save the model
torch.save(alexnet.state_dict(), 'alexnet_no_finetuning_cats_and_dogs.pth')
