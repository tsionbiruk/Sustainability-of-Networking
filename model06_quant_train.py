import torch
import torch.nn as nn
import torch.quantization
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

if __name__=='__main__':
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    print("Finished loading data")

    #quantization aware class
    class CNN6_QAT(nn.Module):
        def __init__(self):
            super(CNN6_QAT, self).__init__()
            self.quant = torch.quantization.QuantStub()  
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, 10)
            self.relu = nn.ReLU()
            self.dequant = torch.quantization.DeQuantStub()  

        def forward(self, x):
            x = self.quant(x)  
            x = self.relu(self.conv1(x))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.relu(self.conv3(x))
            x = self.pool(self.relu(self.conv4(x)))
            x = self.relu(self.conv5(x))
            x = self.pool(self.relu(self.conv6(x)))
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.dequant(x)  
            return x


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("dev:", device)

    # Prepare model for QAT
    model06_quant = CNN6_QAT().to(device)
    model06_quant.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")  # QAT config
    torch.quantization.prepare_qat(model06_quant, inplace=True)  # Convert model to QAT mode

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model06_quant.parameters(), lr=0.001)

    # Training function
    def train_model(model, trainloader, criterion, optimizer, epochs=10):
        print("Initiating Training")
        model.train()
        model.to(device)
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct/total:.2f}%")
        print("Finished Training")


    train_model(model06_quant, trainloader, criterion, optimizer, epochs=10)

    # Convert to fully quantized model after training
    model = torch.quantization.convert(model06_quant, inplace=True)
    print("Model converted to quantized version.")

    torch.save(model.state_dict(), "model06_base_qat.pth")
    print("Quantized model saved as model06_base_qat.pth")




