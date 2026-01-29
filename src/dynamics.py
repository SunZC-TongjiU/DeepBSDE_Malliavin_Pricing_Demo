import numpy as np
import torch

def calculate_C(x):
    """
    Computes the local volatility function C(x) = 10 * sqrt(x).
    Supports both NumPy arrays and PyTorch tensors.
    """
    if torch.is_tensor(x):
        # Add epsilon 1e-6 to avoid gradient explosion at x=0
        return 10.0 * torch.sqrt(torch.maximum(x, torch.tensor(1e-6, device=x.device)))
    else:
        return 10.0 * np.sqrt(np.maximum(x, 1e-6))

def calculate_dC_dx(x):
    """
    Computes the derivative C'(x) = 5 / sqrt(x).
    Used for the Malliavin calculus correction term.
    """
    if torch.is_tensor(x):
        return 5.0 / torch.sqrt(torch.maximum(x, torch.tensor(1e-6, device=x.device)))
    else:
        return 5.0 / np.sqrt(np.maximum(x, 1e-6))