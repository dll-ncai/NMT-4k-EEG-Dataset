# GPU connected
import torch

# Dynamically select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# Test tensor creation on GPU
x = torch.ones(3, 3, device=device)
print("Tensor device:", x.device)  # cuda:0