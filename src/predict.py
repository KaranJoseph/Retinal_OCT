import argparse
import torch
import sys
sys.path.insert(0, 'D:\Github\Federated_Learning\Retinal_OCT\src\centralized')
from train import Retina_Model, train, validate, retina_dataloaders
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def evaluation(dataloader_test, model):
    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloader_test:
          inputs = inputs.to(DEVICE)
          labels = labels.to(DEVICE)

          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
    return labels.tolist(), preds.tolist()

def plot(cm, input_str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix');
    ax.set_xticklabels(labels, rotation=45);
    ax.set_yticklabels(labels, rotation=45);
    fig.savefig(f'{input_str}.png')

def main():
    parser = argparse.ArgumentParser(description='Which model we want to test')
    parser.add_argument('input', metavar='input', type=str, help='model type')
    args = parser.parse_args()

    input_str = args.input
    if input_str == 'centralized':
        PATH = 'D:\Github\Federated_Learning\Retinal_OCT\src\models\centralized\model.pth'
    elif input_str == 'federated':
        PATH = f'D:\Github\Federated_Learning\Retinal_OCT\src\models\\federated\model_round_{num_rounds}.pth'
    else:
        PATH = f'D:\Github\Federated_Learning\Retinal_OCT\src\models\\federated_dp\model_round_{num_rounds}.pth'
    
    model = Retina_Model(DROPOUT, 4).to(DEVICE)
    model.load_state_dict(torch.load(PATH))

    data_dir = 'D:\Github\Federated_Learning\Experiments\RetinalOCT\Data'
    _, _, test_loader = retina_dataloaders(batch_size, input_size, data_dir)
    avg_test_loss = validate(model, DEVICE, test_loader)

    actual, pred = evaluation(test_loader, model) 
    cm = confusion_matrix(actual, pred)
    print(f"Total model accuracy = {accuracy_score(actual, pred)*100:.2f}%")
    report = pd.DataFrame(classification_report(actual, pred, target_names=labels, output_dict=True)).transpose()
    report.to_csv(f'{input_str}.csv')
    return avg_test_loss, cm, input_str

if __name__ == '__main__':
    DROPOUT = 0.2
    batch_size = 32
    input_size = 32
    num_classes = 4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    num_rounds = 10
    _, cm, input_str = main()
    plot(cm, input_str)