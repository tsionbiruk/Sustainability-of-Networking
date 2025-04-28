import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torchvision.models as models
import csv
import os
import subprocess
import time
import pandas as pd

if __name__ == "__main__": 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    #Functions to load base model
    def get_resnet_model(num_classes=100, quantized=False):
        if quantized:
            model = models.quantization.resnet18(pretrained=False, quantize=False)  # Quantization-ready model
        else:
            model = models.resnet18(pretrained=False)  # Standard ResNet18

        model.fc = nn.Linear(512, num_classes)  # Modify final layer for CIFAR-100
        return model.to(device)

    def load_model(model_path, quantized=False):
        model = get_resnet_model(num_classes=100, quantized=quantized)  # Recreate architecture
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
        model.to(device)
        model.eval()  # Set to evaluation mode
        return model

    base_model = load_model("resnet18_full_cifar100.pth", quantized=False)  # Large model

    print("Base Models loaded!")


    # Loading quantized model
    quantized_model = models.quantization.resnet18(weights=None, quantize=False)  # Define model
    quantized_model.fc = nn.Linear(512, 100)  # Adjust output layer for CIFAR-100
    quantized_model.train()
    quantized_model.fuse_model()  # Fuse Conv + BatchNorm layers
    quantized_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(quantized_model, inplace=True)  # Model must be in TRAINING mode here
    quantized_model = torch.quantization.convert(quantized_model)
    quantized_model.load_state_dict(torch.load("resnet18_qat_retrained.pth" , map_location=device))

    print("Quantized Models loaded!")

    # Define parameters
    EXPERIMENTAL_PERIOD = 48  # hours
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

  
    #energy_per_inference_base = 0.3014  # Joules per inference (base model)
    #energy_per_inference_quantized = 0.066  # Joules per inference (quantized model)



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

    POWER_LOG_PATH = "C:\\Program Files\\Intel\\Power Gadget 3.6\\PowerLog3.0.exe"

    def test_model(model, dataset):
      
        model.eval()
        correct, total = 0, 0

        test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)


        accuracy = correct / total if total > 0 else 0
        
        return accuracy
    
    print("Initiating pre-processing for base model")
    random_indices = random.sample(range(len(testset)), SAMPLE_SIZE)
    D_t = Subset(testset, random_indices)

    start_base = time.time()
    test_model(base_model, D_t)
    end_base = time.time()
    buffer = 10
    Base_duration = end_base - start_base - buffer


    start_compressed = time.time()
    test_model(quantized_model, D_t)
    end_compressed = time.time()
    compressed_duration = end_compressed - start_compressed - buffer

    
    
    def start_power_tracking_base(log_file="power_log.csv", resolution_ms=1000, duration_sec= Base_duration):
       
        command = f'"{POWER_LOG_PATH}" -resolution {resolution_ms} -duration {duration_sec} -file {log_file}'
        subprocess.Popen(command, shell=True)


    def start_power_tracking_compressed(log_file="power_log.csv", resolution_ms=1000, duration_sec=compressed_duration):
       
        command = f'"{POWER_LOG_PATH}" -resolution {resolution_ms} -duration {duration_sec} -file {log_file}'
        subprocess.Popen(command, shell=True)
    
    def extract_energy_consumed(log_file):
       
        try:
            df = pd.read_csv(log_file, delimiter=',')  # Read power log

            #print("CSV Columns Found:", df.columns)

            # Expected column name (but verify with print output)
            energy_col = "Cumulative Processor Energy_0(Joules)"
            

            if energy_col in df.columns:
                total_energy = df[energy_col].dropna().iloc[-1]  # Get last recorded value

                # ✅ Delete the file after extracting energy
                os.remove(log_file)
                    
                return total_energy

            print("No matching energy column found in CSV. Check printed columns above.")
            return 0
        except Exception as e:
            print(f"Error reading power log: {e}")
            return 0
        
    
    
    def test_model_base(model, dataset):
      
        model.eval()
        correct, total = 0, 0

        log_file = "power_log.csv"
        start_power_tracking_base(log_file)  # Start tracking power

        test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)


        total_energy = extract_energy_consumed(log_file)  # Get energy usage

        accuracy = correct / total if total > 0 else 0
        energy_per_inference = total_energy / total if total > 0 else 0  # Convert to per inference

        with open("energy_results.csv", "a") as f:
            
            f.write(f"{accuracy},{total_energy},{energy_per_inference}\n")

        return accuracy, energy_per_inference

    def test_model_compressed(model, dataset):
      
        model.eval()
        correct, total = 0, 0

        log_file = "power_log.csv"
        start_power_tracking_compressed(log_file)  # Start tracking power

        test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)


        total_energy = extract_energy_consumed(log_file)  # Get energy usage

        accuracy = correct / total if total > 0 else 0
        energy_per_inference = total_energy / total if total > 0 else 0  # Convert to per inference

        with open("energy_results.csv", "a") as f:
            
            f.write(f"{accuracy},{total_energy},{energy_per_inference}\n")

        return accuracy, energy_per_inference

    
    
    results = []
    print("Initiating Simulation")
    start_time = time.time()

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
        acc_base, energy_base = test_model_base(base_model, D_t)
        acc_quant, energy_quant = test_model_compressed(quantized_model, D_t)

        if model_adaptive == base_model:
            acc_adaptive, energy_adaptive = acc_base, energy_base
        else: 
            acc_adaptive, energy_adaptive = acc_quant, energy_quant
       

        # Store accuracy results
        accuracy_baseline[t] = acc_base
        accuracy_quantized[t] = acc_quant
        accuracy_adaptive[t] = acc_adaptive

        # Store carbon emissions
        carbon_emissions_baseline[t] = compute_carbon_emission(energy_base, current_CI) * SAMPLE_SIZE
        carbon_emissions_quantized[t] = compute_carbon_emission(energy_quant, current_CI) * SAMPLE_SIZE

        carbon_emissions_adaptive[t] = compute_carbon_emission(energy_adaptive, current_CI) * SAMPLE_SIZE

        results.append([t, acc_base, carbon_emissions_baseline[t], acc_quant, carbon_emissions_quantized[t], acc_adaptive, carbon_emissions_adaptive[t]])
        
        round_end_time = time.time()
        round_execution_time = round_end_time - round_start_time

        print(f"Round {t+1}/{NUM_STEPS} | "
            f"Acc (Base): {acc_base:.4f}, CO2: {carbon_emissions_baseline[t]:.4f} g | "
            f"Acc (Quant): {acc_quant:.4f}, CO2: {carbon_emissions_quantized[t]:.4f} g | "
            f"Acc (Adapt): {acc_adaptive:.4f}, CO2: {carbon_emissions_adaptive[t]:.4f} g | "
            f"Round Time: {round_execution_time:.2f} sec")

        if t in [23,47]:
                fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

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
                plt.tight_layout()
                plt.show()


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





