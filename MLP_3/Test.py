import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# ==== Load and preprocess Ames dataset ====
ames = fetch_openml(name="house_prices", as_frame=True)
X_raw = ames.data
y = ames.target.astype(np.float32).values.reshape(-1, 1)

# One-hot encode categorical features
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# Impute and scale
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_encoded)
scaler_X = torch.load("models_AME\scaler_X_reduced.pth", weights_only=False)
scaler_y = torch.load("models_AME\scaler_y_reduced.pth", weights_only=False)
X_scaled = scaler_X.transform(X_imputed)
y_scaled = scaler_y.transform(y)

# Test split
_, X_test_np, _, y_test_np = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ==== Load the model ====
input_size = X_test.shape[1]
model = MLPRegressor(input_size=input_size).to(device)
model.load_state_dict(torch.load("models_AME/pruned.pth"))  # or Struct_Prune.pth
model.eval()

print(f"Model and scalers loaded. Input size: {input_size}")

# ==== Define test function ====
def test(model, dataloader, scaler_y):
    print("Initiating testing...")
    model.eval()
    all_preds, all_targets = [], []
    total_time = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            start_time = time.time()
            outputs = model(inputs)
            total_time += time.time() - start_time
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Combine and inverse scale
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    preds_orig = scaler_y.inverse_transform(all_preds)
    targets_orig = scaler_y.inverse_transform(all_targets)

    # Metrics
    mse = mean_squared_error(targets_orig, preds_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_orig, preds_orig)
    r2 = r2_score(targets_orig, preds_orig)
    avg_inf_time = total_time / len(dataloader)

    print("\n📊 Test Results:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Avg Inference Time per Batch: {avg_inf_time:.6f} sec")

    return mae, rmse, r2, avg_inf_time

# ==== Run test ====
test(model, test_loader, scaler_y)



