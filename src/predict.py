import torch
import sys
from centralized.train import Retina_Model, train, validate, retina_dataloaders

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings("ignore")

PATH = 'D:\Github\Federated_Learning\Retinal_OCT\src\models\centralized'
model = torch.load(PATH)
model.eval()