import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

base_model = MLPRegressor(input_size=8).to(device)
base_model.load_state_dict(torch.load("models/base_mlp.pth"))
base_model.eval()

print("Base model loaded")
#pruned_model = MLPRegressor(input_size=8).to(device)
#pruned_model.load_state_dict(torch.load("models/pruned_mlp.pth"))
#pruned_model.eval()
#print("Pruned model loaded")

scaler_X = torch.load("models/scaler_X.pth", weights_only=False)
scaler_y = torch.load("models/scaler_y.pth", weights_only=False)
print("Scalers loaded")

data = fetch_california_housing()
X_scaled = scaler_X.transform(data.data)
y_scaled = scaler_y.transform(data.target.reshape(-1, 1))
_, X_test, _, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)



test_dataset = TensorDataset(X_test, y_test)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define test function
def test(model, dataloader):
    print("Initiating testing")
    model.eval()
    all_predictions = []
    all_targets = []
    total_time = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            start_time = time.time()
            outputs = model(inputs)

            batch_time = time.time() - start_time
            total_time += batch_time

            all_predictions.extend(outputs.cpu().numpy().flatten())  # Ensure no floating-point issues
            all_targets.extend(targets.cpu().numpy().flatten())

    # Convert lists to NumPy arrays for evaluation
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Compute metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    avg_inference_time = total_time / len(dataloader)

    print(f"Test Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Avg Inference Time per Batch: {avg_inference_time:.6f} sec")

# Run the test function
test(base_model, testloader)


