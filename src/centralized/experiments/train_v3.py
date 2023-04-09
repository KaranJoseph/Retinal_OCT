import torch
import sys
sys.path.insert(0, 'D:\Github\Federated_Learning\Retinal_OCT\src\centralized')
from train import Retina_Model, train, validate, retina_dataloaders
from utils import EarlyStopping

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    data_dir = 'D:\Github\Federated_Learning\Experiments\RetinalOCT\Data'
    batch_size = 32
    input_size = 32
    num_classes = 4

    PATH = 'D:\Github\Federated_Learning\Retinal_OCT\src\models\centralized\model.pth'
    MOMENTUM = 0.9
    LR =  0.0313425891625895
    DROPOUT = 0.2
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Retina_Model(DROPOUT, num_classes).to(DEVICE) #Load base model from centralized model
    criterion = F.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    best_val_loss = float('Inf')
    train_loader, val_loader, _ = retina_dataloaders(batch_size, input_size, data_dir)
    for epoch in range(0, 10):
            avg_train_loss = train(model, DEVICE, train_loader, optimizer, epoch)
            avg_val_loss = validate(model, DEVICE, val_loader)
            
            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss

            # Add custom early stopping
            early_stopping = EarlyStopping()
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                break
            scheduler.step()
    torch.save(model.state_dict(), PATH)
    





