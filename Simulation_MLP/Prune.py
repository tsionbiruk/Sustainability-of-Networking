import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import torch.nn.utils.prune as prune
import os

# ==== Load and preprocess the Ames dataset ====
ames = fetch_openml(name="house_prices", as_frame=True)
X_raw = ames.data
y = ames.target.astype(np.float32).values.reshape(-1, 1)

X_encoded = pd.get_dummies(X_raw, drop_first=True)
X_imputed = SimpleImputer(strategy='mean').fit_transform(X_encoded)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_imputed)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

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

# ==== Load the pretrained base model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_train.shape[1]
model = MLPRegressor(input_size=input_size).to(device)
model.load_state_dict(torch.load("models_AME/base_mlp_reduced.pth"))
print(f"Base model loaded with input size: {input_size}")

# ==== Apply structured pruning ====
def apply_pruning(model, amount=0.3):
    print(f"Applying structured pruning (amount={amount})")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')  # Make pruning permanent
    return model

pruned_model = apply_pruning(model, amount=0.3).to(device)

'''
# ==== Fine-tune the pruned model ====
print("Initiating fine-tuning...")
criterion = nn.MSELoss()
optimizer = optim.Adam(pruned_model.parameters(), lr=0.001)
n_finetune_epochs = 10

for epoch in range(n_finetune_epochs):
    pruned_model.train()
    running_loss = 0.0
    preds, actuals = [], []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = pruned_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds.append(outputs.detach().cpu().numpy())
        actuals.append(targets.detach().cpu().numpy())

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)
    preds_orig = scaler_y.inverse_transform(preds)
    actuals_orig = scaler_y.inverse_transform(actuals)

    mae = mean_absolute_error(actuals_orig, preds_orig)
    rmse = mean_squared_error(actuals_orig, preds_orig)
    r2 = max(0, r2_score(actuals_orig, preds_orig))

    print(f"📘 Epoch [{epoch+1}/{n_finetune_epochs}] - "
          f"Loss: {running_loss/len(train_loader):.4f}, "
          f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2*100:.2f}%")

'''


os.makedirs("models_AME", exist_ok=True)
torch.save(pruned_model.state_dict(), "models_AME/struct_pruned_reduced.pth")
print("Fine-tuned pruned MLP model saved!")



'''
# Post-Pruning Inference Timing Test
def test_inference_time(model, X_test, runs=100):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(X_test_tensor)
    
    total_time = time.time() - start_time
    avg_time = total_time / runs
    print(f"Average inference time per forward pass: {avg_time:.6f} seconds")

print("Testing Base Model Execution Time")
test_inference_time(model, X_test)
print("Testing Pruned Model Execution Time")
test_inference_time(pruned_model, X_test)'
'''
