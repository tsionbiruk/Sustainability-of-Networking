import torch
import torch.nn.utils.prune as prune
import torch.nn as nn


# Define the CNN Model
class CNN6(nn.Module):
    def __init__(self):
        super(CNN6, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.pool(self.relu(self.conv6(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("dev:", device)

# Load trained model
model06_prun_unstruct = CNN6().to(device)
model06_prun_unstruct.load_state_dict(torch.load("model06_base_cifar_trained.pth", map_location=device))
model06_prun_unstruct.eval() 
print("Trained model loaded")


def apply_unstructured_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)  
            prune.remove(module, "weight")  
    return model

# Prune the model
pruned_model = apply_unstructured_pruning(model06_prun_unstruct, amount=0.2)
print("Unstructured pruning applied successfully!")


torch.save(pruned_model.state_dict(), "model06_base_pruned_unstructured.pth")
print("Pruned model saved as model06_base_pruned_unstructured.pth")






