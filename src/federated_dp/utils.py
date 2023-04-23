import torch
import sys
import os
sys.path.insert(0, 'D:\Github\Federated_Learning\Retinal_OCT\src\centralized')
from train import Retina_Model
from torchvision import datasets, transforms

DROPOUT = 0.2
batch_size = 32
input_size = 32
num_classes = 4
data_dir = 'D:\Github\Federated_Learning\Experiments\RetinalOCT\Data'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = 'D:\Github\Federated_Learning\Retinal_OCT\src\models\centralized\model.pth'
model = Retina_Model(DROPOUT, num_classes).to(DEVICE)
model.load_state_dict(torch.load(PATH))

train_transforms = transforms.Compose(
        [transforms.Resize(size=(input_size, input_size)),
        transforms.CenterCrop(size=input_size),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]
        # transforms.Normalize((0.1928,),(0.2022,))] # Calculated mean and std of the train image dataset
    )
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)

def compute_sensitivity(dataset, model):
    max_norm = 0
    for inputs, labels in dataset:
        inputs = torch.tensor(inputs.unsqueeze(0), requires_grad=True)
        output = model(inputs)
        loss = output.max(1)[0].sum()
        loss.backward()
        norm = torch.abs(inputs.grad.data).max()/(input_size*input_size)
        if norm > max_norm:
            max_norm = norm
    return max_norm.item()

sensitivity = compute_sensitivity(train_dataset, model)
print(sensitivity)