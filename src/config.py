import torch

def get_config():
    """
    Returns the configuration dictionary for the Local Stochastic Volatility (LSV) model.
    """
    return {
        # --- Market Parameters ---
        "x0": 100.0,       # Initial Stock Price (S0)
        "sigma0": 0.3,     # Initial Volatility
        "epsilon": 0.1,    # Vol-of-Vol (Perturbation parameter)
        "rho": -0.5,       # Correlation between Brownian motions
        "r": 0.0,          # Risk-free rate
        "T": 0.25,         # Time to maturity (3 months)
        
        # --- Simulation Parameters ---
        "n_steps": 100,    # Number of Euler discretization steps
        "n_paths": 10000, # Number of Monte Carlo paths
        "dt": 0.25 / 100,  # Time step size (T / n_steps)
        
        # --- Deep Learning Parameters ---
        "n_epochs": 30,     # Epoch
        "batch_size": 4096,     #Batch Size

        # --- Evaluation Settings ---
        "strikes": [70, 80, 90, 100, 110, 120, 130], # Strikes to evaluate (ITM to OTM)
        
        # --- Computational Device ---
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }