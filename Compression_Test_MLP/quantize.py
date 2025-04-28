import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from sklearn.metrics import r2_score

# QAT Modules
import torch.ao.quantization as quantization  # Updated import

class HousingDataset(Dataset):
    '''
    Prepare the Housing dataset for regression.
    '''
    def __init__(self, X, y, scale_data=True):
        if scale_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to float32
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # Convert y and reshape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class QAT_MLP(nn.Module):
    '''
    MLP Model modified for Quantization-Aware Training (QAT).
    '''
    def __init__(self, input_size):
        super().__init__()

        # **Add Quantization Stubs**
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),  
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 32),  
            nn.ReLU(),
            nn.Linear(32, 16),  
            nn.ReLU(),
            nn.Linear(16, 1) 
        )

    def forward(self, x):
        x = self.quant(x)  # **Quantization before passing through layers**
        x = self.layers(x)
        x = self.dequant(x)  # **Dequantization after forward pass**
        return x

# Load California Housing dataset
X, y = fetch_california_housing(return_X_y=True)

# Prepare dataset
dataset = HousingDataset(X, y)
trainloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize the QAT-compatible model
qat_model = QAT_MLP(input_size=X.shape[1])  

# Define loss function & optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-3)

# **Step 1: Prepare Model for QAT**
qat_model.train()
qat_model.qconfig = quantization.get_default_qat_qconfig('fbgemm')  # Set QAT config
quantization.prepare_qat(qat_model, inplace=True)  # Prepare for QAT

# **Training Function**
def train_qat(model, optimizer, loss_function, epochs=25):
    model.train()
    print("Initiating QAT training")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_time = 0.0
        all_targets = []
        all_predictions = []

        for inputs, targets in trainloader:
            start_time = time.time()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            all_targets.extend(targets.detach().numpy())  # Collect true values
            all_predictions.extend(outputs.detach().numpy())  # Collect predicted values
            
            total_time += time.time() - start_time

        avg_batch_time = total_time / len(trainloader)
        
        # Convert to NumPy arrays for evaluation
        all_targets = np.array(all_targets).flatten()
        all_predictions = np.array(all_predictions).flatten()

        # Compute accuracy metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))
        r2 = r2_score(all_targets, all_predictions)  

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(trainloader):.4f}, "
              f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}, Avg Batch Time: {avg_batch_time:.4f} sec")

    print("QAT Training finished.")

# Train with QAT
train_qat(qat_model, optimizer, loss_function, 25)

# **Step 2: Convert to Quantized Model**
qat_model.eval()  # Set to evaluation mode
quantization.convert(qat_model, inplace=True)  # Convert to quantized model

# Save the quantized model
#torch.save(qat_model, "quantized_base.pth")  # Save the entire model
print("Fully quantized model saved successfully!")


