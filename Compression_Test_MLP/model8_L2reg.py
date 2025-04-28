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

# Load California Housing dataset
X, y = fetch_california_housing(return_X_y=True)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training (80%), validation + test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Create training dataset & DataLoader
train_dataset = HousingDataset(X_train, y_train, scale_data=False)  # Avoid double standardization
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Save test/validation dataset for later use
np.savez("test_val_data.npz", X_temp=X_temp, y_temp=y_temp)
print("Test/validation dataset saved for separate evaluation.")

# Initialize the model
mlp = MLP(input_size=X.shape[1])  # Dynamically set input size

# Define loss function & optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)

def train(model, optimizer, loss_function, epochs=25):
  
    model.train()
    print("Initiating training")

    for epoch in range(epochs):
        epoch_loss = 0.0
        total_time = 0.0
        all_targets = []
        all_predictions = []

        for batch in train_loader:
            inputs, targets = batch 
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

        avg_batch_time = total_time / len(train_loader)
        
        # Convert to NumPy arrays for evaluation
        all_targets = np.array(all_targets).flatten()
        all_predictions = np.array(all_predictions).flatten()

        # Compute accuracy metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))
        r2 = r2_score(all_targets, all_predictions)  

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, "
              f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}, Avg Batch Time: {avg_batch_time:.4f} sec")

    print("Training finished.")

# Train the model
train(mlp, optimizer, loss_function, epochs=25)

# Save the model weights
torch.save(mlp.state_dict(), "Base_L2reg.pth")
print("Model saved successfully!")
