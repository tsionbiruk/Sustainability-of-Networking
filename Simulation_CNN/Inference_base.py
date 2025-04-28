import torch
import numpy as np
import random
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torchvision.models as models

if __name__ == "__main__": 
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset (CIFAR-100)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    print("Test set loaded!")

    # Simulation Parameters
    experimental_period = 24  # 24 hours
    time_interval = 1  # 1 hour
    num_time_steps = experimental_period // time_interval
    samples_per_step = 1000  # Number of images per hour



    # Define the ResNet18 model (same structure as trained model)
    def get_resnet_model(num_classes=100, quantized=False):
        if quantized:
            model = models.quantization.resnet18(pretrained=False, quantize=False)  # Quantization-ready model
        else:
            model = models.resnet18(pretrained=False)  # Standard ResNet18

        model.fc = nn.Linear(512, num_classes)  # Modify final layer for CIFAR-100
        return model.to(device)

    # Function to Load Model with State Dict
    def load_model(model_path, quantized=False):
        model = get_resnet_model(num_classes=100, quantized=quantized)  # Recreate architecture
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
        model.to(device)
        model.eval()  # Set to evaluation mode
        return model



    # Load both models
    base_model = load_model("resnet18_full_cifar100.pth", quantized=False)  # Large model

    print("Base Models loaded!")

    # Function to Evaluate Model Accuracy
    def evaluate_model(model, dataset, num_samples=1000):
        indices = random.sample(range(len(dataset)), num_samples)  # Random sample
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total  # Accuracy percentage


    
    random_indices = random.sample(range(len(testset)), samples_per_step)
    D_t = Subset(testset, random_indices)

    base_accuracy= evaluate_model(base_model, D_t, num_samples=samples_per_step)
    print(f"Base Model Accuracy: {base_accuracy:.2f}%")


