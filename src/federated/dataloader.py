from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch

import os
data_dir = 'D:\Github\Federated_Learning\Experiments\RetinalOCT\Data' # Change data directory here to replicate code
batch_size = 32
input_size = 32
num_classes = 4


NUM_CLIENTS = 10

def load_data(num_clients):
    """
    Load the train data according to client1 distribution and entire test data
    """
    train_transform = transforms.Compose(
        [transforms.Resize(size=(input_size, input_size)),
         transforms.CenterCrop(size=input_size),
         transforms.RandomAdjustSharpness(sharpness_factor=2),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.1934,),(0.2200,))]
    )

    test_transform = transforms.Compose(
        [transforms.Resize(size=(input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1934,),(0.2200,))])
    
    trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    valset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    train_partition_size = len(trainset) // num_clients
    train_lengths = [train_partition_size] * num_clients
    val_partition_size = len(valset) // num_clients
    val_lengths = [val_partition_size] * num_clients

    train_dataset = random_split(trainset, train_lengths, torch.Generator().manual_seed(42))
    val_dataset = random_split(valset, val_lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for train_ds, val_ds in train_dataset, val_dataset:
        trainloaders.append(DataLoader(train_ds, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(val_ds, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader
