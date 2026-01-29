import numpy as np
from .dynamics import calculate_C, calculate_dC_dx

# --- Method 1: MC with higher-order shceme, using Malliavin IBP
def price_malliavin(cfg, seed=42):
    """
    Computes prices using 1st-order Malliavin Asymptotic Expansion.
    """
    T, x0, sigma0 = cfg["T"], cfg["x0"], cfg["sigma0"]
    epsilon, rho = cfg["epsilon"], cfg["rho"]
    n_paths = cfg["n_paths"]
    
    # 1. Base Gaussian Variable (W_T)
    np.random.seed(seed)
    W_T = np.random.normal(0, np.sqrt(T), n_paths)
    
    # 2. Proxy Model (Linear Approximation)
    vol_val = calculate_C(np.array([x0])).item()
    vol_0 = sigma0 * vol_val
    X_bar = x0 + vol_0 * W_T
    
    # 3. Compute Malliavin Weights
    dc_val = calculate_dC_dx(np.array([x0])).item()
    
    term1 = vol_0 * (sigma0 * dc_val)
    term2 = (epsilon * rho * sigma0) * vol_val
    total_coeff = term1 + term2
    
    # Weight Formula
    weights = 1.0 + (total_coeff / (2 * vol_0 * T)) * W_T * (W_T**2 - 3 * T)
    
    # 4. Pricing
    prices = []
    for K in cfg["strikes"]:
        payoff_proxy = np.maximum(X_bar - K, 0)
        prices.append(np.mean(payoff_proxy * weights))
        
    return prices

# --- Method 2: n-step Eular-Maruyama Scheme and Monto Carlo ---
def price_monte_carlo(cfg, seed=42):
    """
    Computes prices using Euler-Maruyama simulation.
    Ground Truth benchmark.
    """
    T, dt = cfg["T"], cfg["dt"]
    n_paths, n_steps = cfg["n_paths"], cfg["n_steps"]
    x0, sigma0 = cfg["x0"], cfg["sigma0"]
    epsilon, rho = cfg["epsilon"], cfg["rho"]
    sqrt_dt = np.sqrt(dt)
    
    np.random.seed(seed)
    x_mc = np.full(n_paths, x0)
    sigma_mc = np.full(n_paths, sigma0)
    
    # Time Stepping
    for _ in range(n_steps):
        z1 = np.random.normal(0, 1, n_paths)
        z2 = np.random.normal(0, 1, n_paths)
        
        c_val = calculate_C(x_mc)
        
        dw1 = z1 * sqrt_dt
        dw2 = z2 * sqrt_dt
        
        x_mc += sigma_mc * c_val * dw1
        sigma_mc += epsilon * sigma_mc * (rho * dw1 + np.sqrt(1 - rho**2) * dw2)
        
    prices = []
    for K in cfg["strikes"]:
        payoff = np.maximum(x_mc - K, 0)
        prices.append(np.mean(payoff))
        
    return prices

# --- MAIN function ---
def run_benchmark_solvers(cfg):
    print(">>> Starting Benchmark Suite...")
    
    print(f"   Running Monte Carlo ({cfg['n_paths']} paths)...")
    mc_prices = price_monte_carlo(cfg)
    
    print(f"   Running Weak Approxmation by Malliavin IBP...")
    malliavin_prices = price_malliavin(cfg)
    
    return {
        "strikes": cfg["strikes"],
        "mc": mc_prices,
        "malliavin": malliavin_prices
    }