import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Set CPU device
device = torch.device("cpu")
torch.set_num_threads(4)

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load full CIFAR-10 (train + test = 60,000 samples)
full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Combine train and test sets
from torch.utils.data import ConcatDataset
combined_dataset = ConcatDataset([full_dataset, test_dataset])

# Split into 80% train, 10% val, 10% test
total_size = len(combined_dataset)  # 60000
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

trainset, valset, testset = random_split(combined_dataset, [train_size, val_size, test_size])

# DataLoaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Print dataset sizes
print(f"Training set size: {len(trainset)}")
print(f"Validation set size: {len(valset)}")
print(f"Test set size: {len(testset)}")
