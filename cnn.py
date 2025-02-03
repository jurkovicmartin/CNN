import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os

from model import Model


def create_model(epochs: int, error: float, learning_rate: float =0.001, save: bool =False) -> Model:
    """Create and train CNN model.

    Args:
        epochs (int): maximum number of epochs
        error (float): acceptable error value (0-1)
        learning_rate (float, optional): Defaults to 0.001.
        save (bool, optional): saves the model as "model.pth". Defaults to False.

    Returns:
        Model: model
    """
    # Remove previous model
    if os.path.exists("model.pth"):
        os.remove("model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    transform = transforms.Compose([
    # In case of different size
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(root="data/seg_train", transform=transform)
    test_data = datasets.ImageFolder(root="data/seg_test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # print(len(train_data))
    # print(len(train_loader))

    # Getting one batch
    # images, labels = next(iter(train_loader))

    class_names = train_data.classes
    print(f"Classes: {class_names}")


    model = Model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ### TRAINING

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        running_error = float("inf")

        if running_error <= error:
            return

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            # Loss function
            loss = criterion(outputs, labels)
            # Reset gradients
            optimizer.zero_grad()
            # Calculates gradients
            loss.backward()
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        running_error = 1 - (correct / total)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.3f}, Accuracy {correct / total:.3f}")

    ### TESTING

    model.eval()
    correct = 0
    total = 0
    # Skips calculating of the gradients
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {correct / total:.3f}")

    if save:
        torch.save(model.state_dict(), "model.pth")

    return model


def load_model(device: torch.device) -> Model:
    """Loads model with path "model/pth".

    Args:
        device (torch.device): define device to which the model is loaded

    Returns:
        Model: model
    """
    # Model doesn't exist
    if not os.path.exists("model.pth"):
        return None
    
    model = Model()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model