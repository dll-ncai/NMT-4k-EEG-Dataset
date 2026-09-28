import torch 
print("--- PYTORCH CHECK ---") 
print("PyTorch Version:", torch.__version__) 
print("Is PyTorch seeing your RTX 5050?:", torch.cuda.is_available()) 
if torch.cuda.is_available(): 
    print("PyTorch GPU Device:", torch.cuda.get_device_name(0))