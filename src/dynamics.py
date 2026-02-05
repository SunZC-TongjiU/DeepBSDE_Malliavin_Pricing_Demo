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
    
def calculate_d2C_dx2(x):
    """
    Computes the derivative C''(x) = -5 / sqrt(x)**3 /2.
    Used for the Malliavin calculus correction term.
    """
    if torch.is_tensor(x):
        return -2.5 / torch.sqrt(torch.maximum(x, torch.tensor(1e-6, device=x.device)))**3
    else:
        return -2.5 / np.sqrt(np.maximum(x, 1e-6))**3
    
    
    
# deep_bsde/dynamics.py
import torch
import numpy as np

# === Here is for calculating Diffusion Matrix ===
def calculate_V_torch(x, sigma,common, rho, rho_bar):   #common = epsilon *sigma
    
    # Input MUST be tensor ###################
    C_val = 10.0 * torch.sqrt(torch.maximum(x, torch.tensor(1e-6, device=x.device)))
    
    v11 = sigma * C_val
    v12 = torch.zeros_like(x)
    
    v21 = common * rho
    v22 = common * rho_bar
    
    
    row1 = torch.stack([v11, v12], dim=-1)
    row2 = torch.stack([v21, v22], dim=-1)
    return torch.stack([row1, row2], dim=-2)

# === NUMPY ===
def calculate_V_numpy(x, sigma,epsilon, rho, rho_bar):

    C_val = 10.0 * np.sqrt(np.maximum(x, 1e-6))
    
    v11 = sigma * C_val
    v12 = np.zeros_like(x)
    
    common = epsilon * sigma
    v21 = common * rho
    v22 = common * rho_bar
    
    row1 = np.stack([v11, v12], axis=-1)
    row2 = np.stack([v21, v22], axis=-1)
    return np.stack([row1, row2], axis=-2)
    