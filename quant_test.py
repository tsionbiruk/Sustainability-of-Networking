import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    print("Data loaded")


    # Define the CNN Model
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

   
    quantized_model = CNN6_QAT().to(device)

    # Set QAT model to fully quantized mode before loading weights
    quantized_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(quantized_model, inplace=True)  # Prepare for QAT
    torch.quantization.convert(quantized_model, inplace=True)  # Convert to quantized version

    # Load the fully quantized model weights
    quantized_model.load_state_dict(torch.load("model06_base_qat.pth", map_location=device))

    print("Quantized model loaded successfully!")

    def test_model(model, testloader):
        print("Initiating Testing")
        model.eval()
        correct = 0
        total = 0
        inference_times=[]
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                start_time=time.time()
                outputs= model(inputs)
                end_time=time.time()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                inference_times.append(end_time - start_time)

        accuracy = 100 * correct / total
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(f'Accuracy: {accuracy:.2f}%, Avg Inference Time per batch: {avg_inference_time:.6f} seconds')
        print("Finished Testing")

    #test loaded base model
    test_model(quantized_model, testloader)