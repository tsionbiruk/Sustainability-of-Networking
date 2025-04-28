from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from sklearn.metrics import r2_score

class HousingDataset(Dataset):
    '''
    Prepare the Housing dataset for regression.
    '''
    def __init__(self, X, y, scale_data=False):  # Set scale_data=False to avoid double standardization
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to float32
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # Convert y and reshape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP6(nn.Module):
    '''
    Multilayer Perceptron for regression with 6 layers.
    '''
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
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


data = np.load("test_val_data.npz")
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

# Convert test set to DataLoader
test_dataset = HousingDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize two models
input_size = X_test.shape[1]

model2 = MLP6(input_size)

# Load trained weights

model2.load_state_dict(torch.load("Pruned_base_GPU.pth"))

# Move models to evaluation mode

model2.eval()


# Function to evaluate a model
def evaluate(model, test_loader):
    print("Initiating evaluation")
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_targets.extend(targets.numpy())
            all_predictions.extend(outputs.numpy())

    # Convert to NumPy arrays
    all_targets = np.array(all_targets).flatten()
    all_predictions = np.array(all_predictions).flatten()

    # Compute evaluation metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    return {"MSE": mse, "MAE": mae, "R²": r2}


results2 = evaluate(model2, test_loader)


print("\n--- Model Performance ---")
print(f"MSE: {results2['MSE']:.4f}, MAE: {results2['MAE']:.4f}, R²: {results2['R²']:.4f}")

