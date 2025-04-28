import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.ao.quantization as quant
import torch.ao.nn.quantized as nnq
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import csv
import torch.nn.functional as F
import os
import subprocess
import time
import pandas as pd
import torch.nn.qat as nnqat
import torch.quantization as quant

if __name__ == "__main__": 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  
    ])

    # Download CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split training set into train and validation (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    batch_size = 64  

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    #Functions to load base model
    class CNN3(nn.Module):
        def __init__(self):
            super(CNN3, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
            x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
            x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
            x = x.view(-1, 128 * 4 * 4)
            x = self.fc(x)
            return x
    
    class CNNQAT3(nn.Module):
        def __init__(self, qconfig=None):
            super(CNNQAT3, self).__init__()

            # Use default QAT config if none provided
            if qconfig is None:
                qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            self.qconfig = qconfig

            self.quant = quant.QuantStub()
            self.dequant = quant.DeQuantStub()

            self.conv1 = nn.Sequential(
                nnqat.Conv2d(3, 32, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nnqat.Conv2d(32, 64, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nnqat.Conv2d(64, 128, kernel_size=3, padding=1, qconfig=self.qconfig),
                nn.ReLU()
            )

            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = self.quant(x)
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = self.pool(self.conv3(x))
            x = x.reshape(-1, 128 * 4 * 4)
            x = self.fc(x)
            x = self.dequant(x)
            return x

    def load_model(model_path):
        model = CNN3().to(device)  # Recreate architecture
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
        model.eval()
        return model

    base_model = load_model("CNN3_cifar10.pth")  # Large model

    print("Base Models loaded!")


    quantized_model = CNNQAT3()
    quantized_model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(quantized_model, inplace=True)

    quantized_model.load_state_dict(torch.load("CNN3QAT_preconvert.pth", map_location="cpu"))
    quantized_model.eval()


    # ✅ Now convert
    quantized_model = quant.convert(quantized_model.cpu(), inplace=True)
        

    # Define parameters
    EXPERIMENTAL_PERIOD = 24  # hours
    TIME_INTERVAL = 1  # hour per step
    NUM_STEPS = EXPERIMENTAL_PERIOD // TIME_INTERVAL
    SAMPLE_SIZE = 1000  # Number of test images per time step
    PUE = 1.3  # Power Usage Effectiveness

    JOULES_TO_KWH = 3.6e6  # Conversion factor

    # Initialize result storage
    accuracy_baseline = np.zeros(NUM_STEPS)
    accuracy_adaptive = np.zeros(NUM_STEPS)
    accuracy_quantized = np.zeros(NUM_STEPS)
    carbon_emissions_adaptive = np.zeros(NUM_STEPS)
    carbon_emissions_baseline = np.zeros(NUM_STEPS)
    carbon_emissions_quantized = np.zeros(NUM_STEPS)

  
    energy_per_inference_base = 0.3953  # Joules per inference (base model)
    energy_per_inference_quantized = 0.1893  # Joules per inference (quantized model)

    
    def compute_carbon_emission(energy_consumed, carbon_intensity):
        return (energy_consumed / JOULES_TO_KWH) * PUE * carbon_intensity

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

    

    def test_model(model, dataset):
      
        model.eval()
        correct, total = 0, 0

        test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)


        accuracy = correct / total if total > 0 else 0
        
        return accuracy
    
    

    

    
    

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

    
        return scores, lambda_val
    
    results = []
    print("Initiating Simulation")
    start_time = time.time()   

    for t in range(NUM_STEPS):

        round_start_time = time.time()
        hour = t % 24  # Get current hour

        random_indices = random.sample(range(len(testset)), SAMPLE_SIZE)
        D_t = Subset(testset, random_indices)

        if hour in R1 or hour in R3:
            current_CI = 760
        else:
            current_CI= 0

        model_adaptive = quantized_model if (hour in R1 or hour in R3) else base_model

        # Evaluate models
        acc_base = test_model(base_model, D_t)
        acc_quant = test_model(quantized_model, D_t)

        if model_adaptive == base_model:
            acc_adaptive, energy_adaptive = acc_base, energy_per_inference_base
        else: 
            acc_adaptive, energy_adaptive = acc_quant, energy_per_inference_quantized
       

        # Store accuracy results
        accuracy_baseline[t] = acc_base
        accuracy_quantized[t] = acc_quant
        accuracy_adaptive[t] = acc_adaptive

        # Store carbon emissions
        carbon_emissions_baseline[t] = compute_carbon_emission(energy_per_inference_base, current_CI) * SAMPLE_SIZE
        carbon_emissions_quantized[t] = compute_carbon_emission(energy_per_inference_quantized, current_CI) * SAMPLE_SIZE

        carbon_emissions_adaptive[t] = compute_carbon_emission(energy_adaptive, current_CI) * SAMPLE_SIZE

        results.append([t, acc_base, carbon_emissions_baseline[t], acc_quant, carbon_emissions_quantized[t], acc_adaptive, carbon_emissions_adaptive[t]])
        
        round_end_time = time.time()
        round_execution_time = round_end_time - round_start_time

        print(f"Round {t+1}/{NUM_STEPS} | "
            f"Acc (Base): {acc_base:.4f}, CO2: {carbon_emissions_baseline[t]:.4f} g | "
            f"Acc (Quant): {acc_quant:.4f}, CO2: {carbon_emissions_quantized[t]:.4f} g | "
            f"Acc (Adapt): {acc_adaptive:.4f}, CO2: {carbon_emissions_adaptive[t]:.4f} g | "
            f"Round Time: {round_execution_time:.2f} sec")


    # Compute averages
    avg_acc_baseline = np.mean(accuracy_baseline)
    avg_acc_quantized = np.mean(accuracy_quantized)
    avg_acc_adaptive = np.mean(accuracy_adaptive)

    total_emissions_baseline = np.sum(carbon_emissions_baseline)
    total_emissions_quantized = np.sum(carbon_emissions_quantized)
    total_emissions_adaptive = np.sum(carbon_emissions_adaptive)

    print(f"Average Accuracy - Base Model: {avg_acc_baseline:.4f}")
    print(f"Average Accuracy - Quantized Model: {avg_acc_quantized:.4f}")
    print(f"Average Accuracy - Adaptive Strategy: {avg_acc_adaptive:.4f}")
    print(f"Total Carbon Emissions - Base Model: {total_emissions_baseline:.2f} g CO2")
    print(f"Total Carbon Emissions - Quantized Model: {total_emissions_quantized:.2f} g CO2")
    print(f"Total Carbon Emissions - Adaptive Strategy: {total_emissions_adaptive:.2f} g CO2")


    avg_accuracies = {
    "Base Model": avg_acc_baseline,
    "Compressed Model": avg_acc_quantized,
    "Adaptive Strategy": avg_acc_adaptive
    }

    total_co2 = {
        "Base Model": total_emissions_baseline,
        "Compressed Model": total_emissions_quantized,
        "Adaptive Strategy": total_emissions_adaptive
    }

    fig, axs = plt.subplots(1, 2, figsize=(18, 5))  

    # Plot Accuracy Comparison
    axs[0].plot(accuracy_baseline, label="Always Base Model", linestyle='dashed')
    axs[0].plot(accuracy_quantized, label="Always Quantized Model", linestyle='dotted')
    axs[0].plot(accuracy_adaptive, label="Adaptive Strategy", linestyle='solid')
    axs[0].set_xlabel("Time Step (hours)")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Model Accuracy over Time")
    axs[0].legend()

    # Plot Carbon Emissions Comparison
    axs[1].plot(carbon_emissions_baseline, label="Always Base Model", linestyle='dashed')
    axs[1].plot(carbon_emissions_quantized, label="Always Quantized Model", linestyle='dotted')
    axs[1].plot(carbon_emissions_adaptive, label="Adaptive Strategy", linestyle='solid')
    axs[1].set_xlabel("Time Step (hours)")
    axs[1].set_ylabel("Carbon Emissions (g CO2)")
    axs[1].set_title("Carbon Emissions over Time")
    axs[1].legend()

    # Adjust layout and display both plots at the same time
    #axs[2].axis('off')  
    #table_data = [
        #["Base ", f"{total_emissions_baseline:.2f} g CO2"],
        #["Compressed ", f"{total_emissions_quantized:.2f} g CO2"],
        #["Adaptive ", f"{total_emissions_adaptive:.2f} g CO2"]
    #]
    #table = axs[2].table(cellText=table_data, colLabels=["Model", "Total Emissions"], loc='center', cellLoc='center')
    #table.auto_set_font_size(False)
    #table.set_fontsize(9)
    #table.scale(1, 2)
    plt.tight_layout()
    plt.show()



    trade_off_scores, lambda_used = find_index(avg_accuracies, total_co2)

    print(f"Lambda: {lambda_used:.4f}")
    print("Trade-off Scores:", trade_off_scores)

    
    with open("lambda_scores.csv", "a", newline="") as lambda_file:
        writer = csv.writer(lambda_file)
        writer.writerow(["Hour", "Lambda"])
        writer.writerow([hour, lambda_used])

        writer.writerow(["Model", "Trade-off Score"])
        for model, score in trade_off_scores.items():
            writer.writerow([model, score])
        
        writer.writerow([])

    def pick_lambda(avg_accs, total_emissions):
        scores_over_lambda = {}

        max_acc = max(avg_accs.values())
        max_co2 = max(total_emissions.values()) or 1  

        lambda_values = np.arange(0, 2.25, 0.25)  

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

        plt.xlabel("Lambda (λ)")
        plt.ylabel("Trade-off Score")
        plt.title("Model Trade-off Scores vs Lambda")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return scores_over_lambda
    
    pick_lambda(avg_accuracies, total_co2)

    with open("simulation_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time Step", "Accuracy Base", "CO2 Base", "Accuracy Quantized", "CO2 Quantized", "Accuracy Adaptive", "CO2 Adaptive"])
        writer.writerows(results)

        writer.writerow([])
        writer.writerow(["Model", "Trade-off Score"])
        for model, score in trade_off_scores.items():
            writer.writerow([model, score])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time / 60:.2f} min")
    print("Simulation Complete")





