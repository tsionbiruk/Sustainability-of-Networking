import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Set CPU optimizations
    torch.set_num_threads(2)  # Prevent excessive CPU thread contention

    # Transformations for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    print("Finished loading data")

    # Define the CNN Model
    class CNN6(nn.Module):
        def __init__(self):
            super(CNN6, self).__init__()
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

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.relu(self.conv3(x))
            x = self.pool(self.relu(self.conv4(x)))
            x = self.relu(self.conv5(x))
            x = self.pool(self.relu(self.conv6(x)))
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model06 = CNN6().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model06.parameters(), lr=0.001, weight_decay=0.0001)  # L2 Regularization

    # Training function with optimizations
    def train_model(model, trainloader, criterion, optimizer, epochs=10):
        print("Initiating Training")
        model.train()

        best_accuracy = 0.0

        for epoch in range(epochs):
            start_time = time.time()  # Track epoch time
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad(set_to_none=True)  # More memory-efficient

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Track loss and accuracy
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Display epoch stats
            epoch_time = time.time() - start_time
            accuracy = 100 * correct / total

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, "
                  f"Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")

            # Save best model only
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "best_model_L2.pth")

        print("Finished Training")

    # Start training
    train_model(model06, trainloader, criterion, optimizer, epochs=10)

    print("Trained model saved as best_model_L2.pth")
