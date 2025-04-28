import torch
import numpy as np
import random
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torchvision.models as models

if __name__ == "__main__": 
    # ✅ Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load the test dataset (CIFAR-100)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    print("Test set loaded!")

    # ✅ Simulation Parameters
    experimental_period = 24  # 24 hours
    time_interval = 1  # 1 hour
    num_time_steps = experimental_period // time_interval
    samples_per_step = 1000  # Number of images per hour

    
    # ✅ Recreate the quantized ResNet18 model
    quantized_model = models.quantization.resnet18(weights=None, quantize=False)  # Define model
    quantized_model.fc = nn.Linear(512, 100)  # Adjust output layer for CIFAR-100

    # ✅ Set model to TRAINING mode before preparing for QAT
    quantized_model.train()

    # ✅ Fuse Model (Required for QAT)
    quantized_model.fuse_model()  # Fuse Conv + BatchNorm layers

    # ✅ Set QAT Configuration
    quantized_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

    # ✅ Prepare Model for QAT
    torch.quantization.prepare_qat(quantized_model, inplace=True)  # Model must be in TRAINING mode here

    # ✅ Convert model to quantized format
    quantized_model = torch.quantization.convert(quantized_model)

    # ✅ Load the quantized model's state_dict
    quantized_model.load_state_dict(torch.load("resnet18_qat_retrained.pth"))

    print("Quantized Models loaded!")

    # ✅ Function to Evaluate Model Accuracy
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


    # ✅ Generate a consistent dataset sample for fairness
    random_indices = random.sample(range(len(testset)), samples_per_step)
    D_t = Subset(testset, random_indices)

    Quantized_accuracy= evaluate_model(quantized_model, D_t, num_samples=samples_per_step)
    print(f"Base Model Accuracy: {Quantized_accuracy:.2f}%")