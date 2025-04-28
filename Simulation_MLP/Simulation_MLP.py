import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import pandas as pd
import time
import os
from sklearn.datasets import fetch_openml
import random
import csv

# ==== Load models and scalers ====
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

# Load dataset and scalers
ames = fetch_openml(name="house_prices", as_frame=True)
X_raw = ames.data
y = ames.target.astype(np.float32).values.reshape(-1, 1)

X_encoded = pd.get_dummies(X_raw, drop_first=True)
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_encoded)
scaler_X = torch.load("models_AME/scaler_X_reduced.pth", weights_only=False)
scaler_y = torch.load("models_AME/scaler_y_reduced.pth", weights_only=False)
X_scaled = scaler_X.transform(X_imputed)
y_scaled = scaler_y.transform(y)

_, X_test_np, _, y_test_np = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load models
input_size = X_test.shape[1]
pruned_model = MLPRegressor(input_size=input_size).to(device)
pruned_model.load_state_dict(torch.load("models_AME/struct_pruned_reduced.pth"))
pruned_model.eval()

base_model = MLPRegressor(input_size=input_size).to(device)
base_model.load_state_dict(torch.load("models_AME/base_mlp_reduced.pth"))
base_model.eval()

print(f"Model and scalers loaded. Input size: {input_size}")

# ==== Simulation settings ====
EXPERIMENTAL_PERIOD = 48
TIME_INTERVAL = 1
NUM_STEPS = EXPERIMENTAL_PERIOD // TIME_INTERVAL

SAMPLE_SIZE = 200
PUE = 1.3
JOULES_TO_KWH = 3.6e6
energy_per_inference_base = 1.44
energy_per_inference_pruned = 1.23
#CO2_per_unit_energy = 0.2

# Define energy schedule (Brown: 00:01-06:00, 12:01-18:00; Green: 06:01-12:00, 18:01-00:00)
R1 = list(range(0, 6))  # Brown energy _ coal
R3 = list(range(12, 18)) # brown energy _ coal
R2 = list(range(6, 12))  # Green energy _ solar
R4 = list(range(18, 24))  # Green energy _ wind

# Define CI schedual (Direct emission)
CI_R1 = 760
CI_R2 = 0
CI_R3 = 760
CI_R4 = 0

accuracy_baseline = np.zeros(NUM_STEPS)
accuracy_adaptive = np.zeros(NUM_STEPS)
accuracy_pruned = np.zeros(NUM_STEPS)
carbon_emissions_adaptive = np.zeros(NUM_STEPS)
carbon_emissions_baseline = np.zeros(NUM_STEPS)
carbon_emissions_pruned = np.zeros(NUM_STEPS)

#energy_per_inference_base=1.66 #fine tuned
#energy_per_inference_pruned=1.56 #fine tuned

#not fine tuned
energy_per_inference_base=1.7435
energy_per_inference_pruned=1.5604


def compute_carbon_emission(energy_consumed, carbon_intensity):
    return (energy_consumed / JOULES_TO_KWH) * PUE * carbon_intensity

def test_model(model, dataset, targets_tensor):
    model.eval()
    predictions = []
    actuals = []

    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        start_idx = 0
        for inputs in test_loader:
            batch_size = inputs.size(0)
            end_idx = start_idx + batch_size

            inputs = inputs.to(device)
            targets = targets_tensor[start_idx:end_idx].to(device)

            outputs = model(inputs)

            predictions.append(outputs.cpu())
            actuals.append(targets.cpu())

            start_idx = end_idx

  
    predictions = torch.cat(predictions).squeeze().numpy()
    actuals = torch.cat(actuals).squeeze().numpy()

    
    r2 = r2_score(actuals, predictions)
    #r2_percentage = max(0, r2)  # Clamp to 0 minimum

    return r2

      




def find_index(avg_accs, total_emissions):
                
    acc_b = avg_accs["Base Model"]
    co2_b = total_emissions["Base Model"]
    acc_q = avg_accs["Compressed Model"]
    co2_q = total_emissions["Compressed Model"]

    acc_diff = acc_b - acc_q
    co2_diff = co2_b - co2_q

    if acc_diff == 0:
        raise ValueError("Base and compressed models have the same accuracy — ACTI is undefined.")
    acti_bq = co2_diff / acc_diff
    lambda_val = 1 / acti_bq if acti_bq != 0 else float("inf")

    # Normalize
    max_acc = max(avg_accs.values())
    max_co2 = max(total_emissions.values()) or 1  # Avoid division by zero

    scores = {}
    for name in avg_accs:
        norm_acc = avg_accs[name] / max_acc
        norm_co2 = total_emissions[name] / max_co2
        scores[name] = norm_acc - lambda_val * norm_co2


    return scores, lambda_val, acti_bq


results = []
print("Initiating Simulation")
start_time = time.time()

for t in range(NUM_STEPS):
    round_start_time = time.time()
    hour = t % 24  

    idx = random.sample(range(len(X_test)), SAMPLE_SIZE)
    D_t = Subset(X_test, idx)
    targets = y_test[idx]  


    current_CI = 760 if hour in R1 or hour in R3 else 0

    model_adaptive = pruned_model if hour in R1 or hour in R3 else base_model

    # Evaluate models
    acc_base = test_model(base_model, D_t, targets)
    acc_pruned = test_model(pruned_model, D_t, targets)
    
    if model_adaptive == base_model:
            acc_adaptive, energy_adaptive_used = acc_base, energy_per_inference_base
    else: 
        acc_adaptive, energy_adaptive_used = acc_pruned, energy_per_inference_pruned

   
    accuracy_baseline[t] = acc_base
    accuracy_pruned[t] = acc_pruned
    accuracy_adaptive[t] = acc_adaptive

    # Store carbon emissions per model
    carbon_emissions_baseline[t] = compute_carbon_emission(energy_per_inference_base, current_CI) * SAMPLE_SIZE
    carbon_emissions_pruned[t] = compute_carbon_emission(energy_per_inference_pruned, current_CI) * SAMPLE_SIZE
    carbon_emissions_adaptive[t] = compute_carbon_emission(energy_adaptive_used, current_CI) * SAMPLE_SIZE

    # Store everything in results array
    results.append([
        t,
        acc_base, carbon_emissions_baseline[t],
        acc_pruned, carbon_emissions_pruned[t],
        acc_adaptive, carbon_emissions_adaptive[t]
    ])

    round_end_time = time.time()
    round_execution_time = round_end_time - round_start_time


    print(f"Round {t+1}/{NUM_STEPS} | "
          f"r^2 (Base): {acc_base:.4f}, CO2: {carbon_emissions_baseline[t]:.4f} g | "
          f"r^2 (Pruned): {acc_pruned:.4f}, CO2: {carbon_emissions_pruned[t]:.4f} g | "
          f"r^2 (Adapt): {acc_adaptive:.4f}, CO2: {carbon_emissions_adaptive[t]:.4f} g | "
          f"Round Time: {round_execution_time:.2f} sec")
    
    if t in [23, 47]: 
        
        # Compute averages
        avg_acc_baseline = np.mean(accuracy_baseline)
        avg_acc_pruned = np.mean(accuracy_pruned)
        avg_acc_adaptive = np.mean(accuracy_adaptive)

        total_emissions_baseline = np.sum(carbon_emissions_baseline)
        total_emissions_pruned = np.sum(carbon_emissions_pruned)
        total_emissions_adaptive = np.sum(carbon_emissions_adaptive)

        print(f"Summary for hour:{t}")
        print(f"Average Accuracy - Base Model: {avg_acc_baseline:.4f}")
        print(f"Average Accuracy - pruned Model: {avg_acc_pruned:.4f}")
        print(f"Average Accuracy - Adaptive Strategy: {avg_acc_adaptive:.4f}")
        print(f"Total Carbon Emissions - Base Model: {total_emissions_baseline:.2f} g CO2")
        print(f"Total Carbon Emissions - pruned Model: {total_emissions_pruned:.2f} g CO2")
        print(f"Total Carbon Emissions - Adaptive Strategy: {total_emissions_adaptive:.2f} g CO2")

        avg_accuracies = {
        "Base Model": max(0, min(avg_acc_baseline, 1)),
        "Compressed Model": max(0, min(avg_acc_pruned, 1)),
        "Adaptive Strategy": max(0, min(avg_acc_adaptive, 1))
        }

        total_co2 = {
            "Base Model": total_emissions_baseline,
            "Compressed Model": total_emissions_pruned,
            "Adaptive Strategy": total_emissions_adaptive
        }

        # Setting up the figure and axes
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [3, 3, 1]})

        # Plotting accuracy comparison
        axs[0].plot(accuracy_baseline, label="Always Base Model", linestyle='dashed')
        axs[0].plot(accuracy_pruned, label="Always Pruned Model", linestyle='dotted')
        axs[0].plot(accuracy_adaptive, label="Adaptive Strategy", linestyle='solid')
        axs[0].set_xlabel("Time Step (hours)")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_title("Model Accuracy over Time")
        axs[0].legend()

        # Plotting carbon emissions comparison
        axs[1].plot(carbon_emissions_baseline, label="Always Base Model", linestyle='dashed')
        axs[1].plot(carbon_emissions_pruned, label="Always Pruned Model", linestyle='dotted')
        axs[1].plot(carbon_emissions_adaptive, label="Adaptive Strategy", linestyle='solid')
        axs[1].set_xlabel("Time Step (hours)")
        axs[1].set_ylabel("Carbon Emissions (g CO2)")
        axs[1].set_title("Carbon Emissions over Time")
        axs[1].legend()

        # Creating a table for total emissions
        axs[2].axis('off')  
        table_data = [
            ["Base ", f"{total_emissions_baseline:.2f} g CO2"],
            ["Compressed ", f"{total_emissions_pruned:.2f} g CO2"],
            ["Adaptive ", f"{total_emissions_adaptive:.2f} g CO2"]
        ]
        table = axs[2].table(cellText=table_data, colLabels=["Model", "Total Emissions"], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        plt.tight_layout()
        plt.show()
        trade_off_scores, lambda_used, acti_value = find_index(avg_accuracies, total_co2)

        print(f"Lambda: {lambda_used:.4f}")
        print("Trade-off Scores:", trade_off_scores)


        # Save lambda and trade-off scores to a separate CSV
        with open("lambda_scores.csv", "a", newline="") as lambda_file:
            writer = csv.writer(lambda_file)
            writer.writerow(["Hour", "Lambda (1/ACTI)"])
            writer.writerow([hour, lambda_used])

            writer.writerow(["Model", "Trade-off Score"])
            for model, score in trade_off_scores.items():
                writer.writerow([model, score])
            
            writer.writerow([])  
 
        
        def pick_lambda(avg_accs, total_emissions):
            scores_over_lambda = {}

            max_acc = max(avg_accs.values())
            max_co2 = max(total_emissions.values()) or 1  

            lambda_values = np.arange(0, 4.25, 0.25)  

            # Initialize score lists for each model
            for name in avg_accs:
                scores_over_lambda[name] = []

            # Compute scores for each lambda
            for lambda_val in lambda_values:
                for name in avg_accs:
                    norm_acc = avg_accs[name] / max_acc
                    norm_co2 = total_emissions[name] / max_co2
                    score = norm_acc - lambda_val * norm_co2
                    scores_over_lambda[name].append(score)

            # Plot the results
            plt.figure(figsize=(10, 5))
            for name, scores in scores_over_lambda.items():
                plt.plot(lambda_values, scores, label=name)

            plt.xlabel("λ")
            plt.ylabel("Trade-off Score")
            plt.title("Best model selection based on λ")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            return scores_over_lambda
        
        pick_lambda(avg_accuracies, total_co2)

        # Save results to CSV
        with open(f"simulation_results_hour:{hour}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time Step", "Compressed Base", "CO2 Base", "Accuracy Compressed", "CO2 Compressed", "Accuracy Adaptive", "CO2 Adaptive"])
            writer.writerows(results)

            writer.writerow([])
            writer.writerow(["Model", "Trade-off Score"])
            for model, score in trade_off_scores.items():
                writer.writerow([model, score])


end_time = time.time()
execution_time = end_time - start_time
print(f"Total Simulation Time: {execution_time / 60:.2f} min")
print("Simulation Complete")



