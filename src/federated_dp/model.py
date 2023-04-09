import torch
from dataloader import load_data
import sys
sys.path.insert(0, 'D:\Github\Federated_Learning\Retinal_OCT\src\centralized')
from train import Retina_Model

from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import flwr as fl

NUM_CLIENTS = 4
MOMENTUM = 0.9
LR =  0.0313425891625895
DROPOUT = 0.2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Retina_Model(DROPOUT, 4).to(DEVICE) #Load base model from centralized model

trainloaders, valloader, testloader = load_data(NUM_CLIENTS)
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def train(model, trainloader, epochs: int):
    """Train the modelwork on the training set."""
    criterion = F.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(model, testloader):
    """Evaluate the modelwork on the entire test set."""
    criterion = F.nll_loss
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    print(f"Test loss: {loss}, accuracy: {accuracy}")
    return loss, accuracy


