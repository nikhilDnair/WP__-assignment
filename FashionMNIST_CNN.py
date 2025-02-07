import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the CNN model (same as MNIST_CNN.py)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for FashionMNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
fashion_mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
fashion_mnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

test_loader = DataLoader(fashion_mnist_testset, batch_size=64, shuffle=False)

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier()

# Load the pre-trained weights (MNIST model)
model.load_state_dict(torch.load("./ModelFiles/model.pt"))

# Move the model to device
model.to(device)

# Set the model to evaluation mode
model.eval()

# Evaluate the model on FashionMNIST
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on FashionMNIST test data using pre-trained MNIST model: {accuracy:.2f}%")
