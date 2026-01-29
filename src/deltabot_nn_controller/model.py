import numpy as np
import matplotlib.pyplot as plt
import torch

print("torch version: ", torch.__version__)
print(f"is cuda available? {torch.cuda.is_available()}")
print(f"using {torch.cuda.get_device_name(0)}")
device = "cuda" if torch.cuda.is_available() else "cpu"