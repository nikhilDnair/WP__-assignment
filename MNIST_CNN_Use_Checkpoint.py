import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(self._get_conv_output_size(), 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_conv_output_size(self):
        # Simulate a single forward pass to determine the output size of conv layers
        x = torch.zeros(1, 1, 28, 28)  # Size of input image (28x28x1 for MNIST)
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # Pooling after conv1
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # Pooling after conv2
        return x.numel()  # Get the number of elements in the output tensor after conv layers

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define transformations and load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Create model and optimizer
model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Function to load checkpoint
def load_checkpoint(filename="./checkpoints/checkpoint.pt"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["last_epoch"]
    loss = checkpoint["last_loss"]
    print(f"Checkpoint loaded from epoch {epoch}, loss: {loss:.4f}")
    return epoch, loss

def save_checkpoint(epoch, model, optimizer, loss, filename="./checkpoints/checkpoint.pt"):
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    checkpoint = {
        "last_epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "last_loss": loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")


# Load checkpoint and resume training
checkpoint_file = './checkpoints/checkpoint.pt'
start_epoch = 0
if os.path.exists(checkpoint_file):
    start_epoch, last_loss = load_checkpoint(checkpoint_file)
else:
    print("No checkpoint found, starting from scratch.")

# Resume training from the checkpoint
num_epochs = 10
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Optionally save the checkpoint again after every epoch or based on conditions
    save_checkpoint(epoch, model, optimizer, avg_loss)

print("Resumed training finished.")
