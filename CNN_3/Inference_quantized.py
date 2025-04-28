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
import torch.nn.qat as nnqat
import torch.quantization as quant

if __name__ == '__main__':
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


    class CNNQAT3(nn.Module):
        def __init__(self, qconfig=None):
            super(CNNQAT3, self).__init__()

            # Use default QAT config if none provided
            if qconfig is None:
                qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            self.qconfig = qconfig

            self.quant = quant.QuantStub()
            self.dequant = quant.DeQuantStub()

            self.conv1 = nn.Sequential(
                nnqat.Conv2d(3, 32, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nnqat.Conv2d(32, 64, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nnqat.Conv2d(64, 128, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )

            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = self.quant(x)
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = self.pool(self.conv3(x))
            x = x.reshape(-1, 128 * 4 * 4)
            x = self.fc(x)
            x = self.dequant(x)
            return x
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("dev:", device)

   
    quantized_model = CNNQAT3()
    quantized_model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(quantized_model, inplace=True)

    quantized_model.load_state_dict(torch.load("CNN3QAT_preconvert.pth", map_location="cpu"))
    quantized_model.eval()

    # ✅ Now convert
    quantized_model = quant.convert(quantized_model.cpu(), inplace=True)

    def test_model(model, dataset):
        print("Initiating inference")
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
        print("Finished Testing")
        return accuracy

    random_indices = random.sample(range(len(testset)), 1000)
    D_t = Subset(testset, random_indices)

    base_accuracy= test_model(quantized_model, D_t)
    