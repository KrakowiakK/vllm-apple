import torch
print("Imported torch")
try:
    if torch.backends.mps.is_available():
        print("MPS is available")
        x = torch.ones(1, device="mps")
        print("Tensor created")
        print(x)
    else:
        print("MPS NOT available")
except Exception as e:
    print(f"Error: {e}")
