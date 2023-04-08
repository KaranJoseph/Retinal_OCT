"""
This script contains the training code for Retinal OCT dataset using PyTorch
Optuna - hyperparameter optimization
mlflow - experiment tracking

**Purpose of the script is to build a baseline model after a moderate search space exploration**

Future improvements:
    - Add model save and load features with optuna -> helps save weights and models
    - Connect optuna to SQL (MySQL/Postgres) to pause, save, and resume trials
    - Parallel/distributed training by connecting optuna to SQL db
    - Look into how optuna can be used with pytorch/tensorflow to modify model architecture
    - explore mflow projects and mlflow models for ease of deployment and serving

Reference:
 - https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
 - https://github.com/StefanieStoppel/pytorch-mlflow-optuna/blob/tutorial-basics/mlflow-optuna-pytorch.ipynb
 - https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html#activating-pruners
 - https://medium.com/analytics-vidhya/predict-retinal-disease-with-cnn-retinal-oct-images-dataset-6df09cb50206
 - https://www.cs.toronto.edu/~lczhang/360/lec/w04/convnet.html
 - https://github.com/optuna/optuna-examples/blob/main/kubernetes/mlflow/pytorch_lightning_distributed.py
"""



from __future__ import print_function
import mlflow
import os
import tempfile
import torch
import optuna

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from mlflow import pytorch
from pprint import pformat

from logger import set_logger
import time
from utils import EarlyStopping


import os
data_dir = 'D:\Github\Federated_Learning\Experiments\RetinalOCT\Data' # Change data directory here to replicate code
batch_size = 32
input_size = 32
num_classes = 4

# Set Mlflow experiment name
experiment_name = f"Retinal Experiment"
experiment_id = mlflow.create_experiment(experiment_name)
tracking_uri = mlflow.get_tracking_uri()

## Setup logger
logger = set_logger()


class Retina_Model(nn.Module):
    def __init__(self, dropout=0.0):
        super(Retina_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # First convolution layer
        x = self.pool(F.relu(self.conv2(x))) # 2nd convolution layer + maxpool
        x = self.dropout1(x) # Add dropouts
        x = torch.flatten(x, 1) # Flatten before connecting to FC layer
        x = F.relu(self.fc1(x)) # First fully connected layer
        x = self.dropout2(x)
        x = self.fc2(x) # 2nd fully connected layer -> output is num_classes
        output = F.log_softmax(x, dim=1) # log transform before calculating loss function
        return output

# Training loop
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_set_size = len(train_loader.dataset)
    num_batches = len(train_loader)
    train_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label) # null loss -> same as cross_entropy loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            batch_size = len(data)
            logger.info(f"Train Epoch: {epoch} [{batch_idx * batch_size}/{train_set_size}"
                        f"({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.6f}") #Monitor status of training 
    avg_train_loss = train_loss / num_batches
    return avg_train_loss

# Testing loop
def validate(model, device, val_loader):
    model.eval()
    val_set_size = len(val_loader.dataset)
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, label, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    val_loss /= val_set_size

    logger.info(f"Test set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{val_set_size} "
                f"({100. * correct / val_set_size:.0f}%)\n")
    return val_loss


def search_space(trial):
    """
    Define parameter search space - Only tuning lr, dropout
    Optimizer is kept as SGD to reduce training time
    """
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.6, step=0.1)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD"])
    momentum = trial.suggest_float("momentum", 0.81, 0.99, step=0.03)

    logger.info(f"Suggested hyperparameters: \n{pformat(trial.params)}")
    return lr, dropout, optimizer_name, momentum

def retina_dataloaders(batch_size=32):
    """
    load train, validation and test sets from data_dir 
    perform transformations to augment train data and reduce size for faster training
    """
    data_transforms = {}
    data_transforms['train'] = transforms.Compose(
        [transforms.Resize(size=(input_size, input_size)),
        transforms.CenterCrop(size=input_size),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1934,),(0.2200,))] # Calculated mean and std of the train image dataset
    )
    
    # First experiment was run with normalization set to (0.1934,),(0.2200,) -> resize = 500
    # If time permits run with mean and std set to (0.1928,),(0.2022,) -> resize = 32 
    # Not expecting big changes as values are close

    data_transforms['val'] = transforms.Compose(
        [transforms.Resize(size=(input_size, input_size)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1934,),(0.2200,))] # Calculated mean and std of the train image dataset
    )


    retina_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    retina_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms['val'])
    retina_test = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(retina_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(retina_val, batch_size=1000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(retina_test, batch_size=1000, shuffle=True)

    return train_loader, val_loader, test_loader

def objective(trial):
    """
    Optuna hyperparamter tuning (Bayesian TPE)
    """
    logger.info("\n********************************\n")
    start_time = time.time()
    best_val_loss = float('Inf')
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        logger.info("Active Run ID: %s\n" % (run.info.run_uuid))
        lr, dropout, optimizer_name, momentum = search_space(trial)
        mlflow.log_params(trial.params)

        # Use CUDA if GPU is available and log device as param using mlflow
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)

        # Initialize Model
        model = Retina_Model(dropout=dropout).to(device)

        # Choose optmizer 
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if optimizer_name == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        
        train_loader, val_loader, _ = retina_dataloaders(batch_size)
        
        # Train and validation loop -> Choosing 5 epochs to reduce training time
        for epoch in range(0, 10):
            avg_train_loss = train(model, device, train_loader, optimizer, epoch)
            avg_val_loss = validate(model, device, val_loader)
            
            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss

            mlflow.log_metric("avg_train_losses", avg_train_loss, step=epoch)
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
            
            #Add optuna trial pruning based on val_loss
            trial.report(avg_val_loss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Add custom early stopping
            early_stopping = EarlyStopping()
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                break
            scheduler.step()

    logger.info(f'Trial Complete at epoch {epoch}')
    logger.info(f'--------------{(time.time()-start_time)/60:.3f} minutes------------')
    return best_val_loss

def main():
    start_time = time.time()
    study = optuna.create_study(study_name="retina_ocl-mlflow-optuna", direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10)

    # Log the trial results
    logger.info("\n++++++++++++++++++++++++++++++++++\n")
    logger.info("Trial results: ")
    logger.info("  Number of finished trials: ", len(study.trials))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Trial number: ", trial.number)
    logger.info("  Loss (trial value): ", trial.value)

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
    
    logger.info(f"Experiment total time: {(time.time()-start_time)/60:.3f} minutes")


if __name__ == '__main__':
    main()
