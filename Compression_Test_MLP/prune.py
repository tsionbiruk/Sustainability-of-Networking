import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_california_housing 
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import r2_score
import numpy as np
import torch.nn.utils.prune as prune

class HousingDataset(Dataset):

    def __init__(self, X, y, scale_data=True):
        if scale_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
         
        self.X = torch.tensor(X, dtype=torch.float32)  
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # Convert y and reshape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression with 8 layers.
    '''
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # Output layer with linear activation
        )

    def forward(self, x):
        return self.layers(x)


#torch.manual_seed(42)

# Load California Housing dataset (instead of Boston)
X, y = fetch_california_housing(return_X_y=True)  # 8 features

# Prepare dataset
dataset = HousingDataset(X, y)
trainloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize the model
mlp = MLP(input_size=X.shape[1])  # Dynamically set input size


mlp.load_state_dict(torch.load("Base_L2reg.pth"))
mlp.eval() 
print("Trained model loaded")


def apply_unstructured_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  
            prune.l1_unstructured(module, name="weight", amount=amount)  
            prune.remove(module, "weight") 

    return model


pruned_model = apply_unstructured_pruning(mlp, amount=0.3)
print("Unstructured pruning applied.")


torch.save(pruned_model.state_dict(), "Pruned_l2reg.pth")
print("Pruned model saved.")


