import torch
import torch.nn as nn
import torch.optim as optim
import time
import torchvision
import random
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  
    ])

    # Download CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split training set into train and validation (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    batch_size = 64  

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    #Functions to load base model
    class CNN3(nn.Module):
        def __init__(self):
            super(CNN3, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
            x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
            x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
            x = x.view(-1, 128 * 4 * 4)
            x = self.fc(x)
            return x



    def load_model(model_path):
        model = CNN3().to(device)  # Recreate architecture
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
        model.eval()
        return model

    base_model = load_model("CNN3_cifar10.pth")  # Large model

    print("Base Models loaded!")


    def test_model(model, dataset):
      
        model.eval()
        correct, total = 0, 0

        test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)


        accuracy = correct / total if total > 0 else 0
        
        

        accuracy = 100 * correct / total
        
        print(f'Accuracy: {accuracy:.2f}%')
        
        return accuracy

    random_indices = random.sample(range(len(testset)), 1000)
    D_t = Subset(testset, random_indices)

    base_accuracy= test_model(base_model, D_t)
    print(f"Base Model Accuracy: {base_accuracy:.2f}%")


