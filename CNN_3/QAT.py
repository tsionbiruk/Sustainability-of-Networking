import torch
import torch.nn as nn
import torch.quantization
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import torchvision
import torch.nn.qat as nnqat
import torch.quantization as quant

if __name__ == '__main__':
    torch.set_num_threads(4)

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR-10
    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([full_dataset, test_dataset])

    total_size = len(combined_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    trainset, valset, testset = random_split(combined_dataset, [train_size, val_size, test_size])

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    print("Data loaded")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    class CNNQAT3(nn.Module):
        def __init__(self, qconfig=None):
            super(CNNQAT3, self).__init__()

            if qconfig is None:
                qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            self.qconfig = qconfig

            self.quant = quant.QuantStub()
            self.dequant = quant.DeQuantStub()

            self.conv1 = nn.Sequential(
                nnqat.Conv2d(3, 16, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nnqat.Conv2d(16, 32, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nnqat.Conv2d(32, 64, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )

            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 4 * 4, 10)

        def forward(self, x):
            x = self.quant(x)
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = self.pool(self.conv3(x))
            x = x.reshape(-1, 64 * 4 * 4)
            x = self.fc(x)
            x = self.dequant(x)
            return x

    # Initialize model
    model = CNNQAT3()

    # Fuse modules
    torch.quantization.fuse_modules(
        model,
        [["conv1.0", "conv1.1"], ["conv2.0", "conv2.1"], ["conv3.0", "conv3.1"]],
        inplace=True
    )

    # Set QAT config
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')

    # Prepare model for QAT
    quant.prepare_qat(model, inplace=True)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training loop
    def train(model, trainloader, testloader, epochs=15):
        print("Initiating training")
        model.train()
        model.to(device)

        for epoch in range(epochs):
            start = time.time()
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

            # Evaluate on test set
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            model.train()

            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Loss: {running_loss/len(trainloader):.4f} | "
                  f"Train Acc: {100*correct/total:.2f}% | "
                  f"Test Acc: {100*test_correct/test_total:.2f}% | "
                  f"Time: {time.time() - start:.2f}s")

    # Start training
    train(model, trainloader, testloader, epochs=15)

    # Save QAT model before convert
    torch.save(model.state_dict(), "CNN3QAT_preconvert.pth")
    print("Saved QAT model before conversion.")

    # Convert to quantized version
    model.eval()
    quantized_model = quant.convert(model.cpu(), inplace=False)

    # Save final quantized model
    torch.save(quantized_model.state_dict(), "CNN3QAT_quantized.pth")
    print("Final quantized model saved.")

        
    