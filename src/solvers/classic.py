import numpy as np
from ..dynamics import calculate_C, calculate_dC_dx

# --- Method 1: MC with higher-order shceme, using Malliavin IBP
def price_malliavin(cfg, seed=42):
    """
    Computes prices using 1st-order Malliavin Asymptotic Expansion.
    """
    T, dt = cfg["T"], cfg["dt"]
    n_paths, n_steps = cfg["n_paths"], cfg["n_steps"]
    x0, sigma0 = cfg["x0"], cfg["sigma0"]
    epsilon, rho = cfg["epsilon"], cfg["rho"]
    sqrt_dt = np.sqrt(dt)
    
    
    # 1. Base Gaussian Variable (W_T)
    np.random.seed(seed)
    
    x_em = np.full(n_paths, x0)
    sigma_em = np.full(n_paths, sigma0)
    weights = np.full(n_paths, 1)
    
    # Time Stepping
    for _ in range(n_steps):
        
        #Eular-Maruyama Scheme
        z1 = np.random.normal(0, 1, n_paths)
        z2 = np.random.normal(0, 1, n_paths)
        
        c_val = calculate_C(x_em)
        dc_val = calculate_dC_dx(x_em)
        
        dw1 = z1 * sqrt_dt
        dw2 = z2 * sqrt_dt
        
        x_next += sigma_em * c_val * dw1
        sigma_next += epsilon * sigma_em * (rho * dw1 + np.sqrt(1 - rho**2) * dw2)
        
        #Compute Malliavin Weights
        H_11 = dw1*(dw1**2-3*dt)/(2*dt*dt)*(sigma_em*dc_val + epsilon*rho)
        
        # Weight Formula
        weights *= (1.0 + H_11)
        
        #Update Data
        x_em = x_next
        sigma_em = sigma_next
    
    # 4. Pricing
    prices = []
    for K in cfg["strikes"]:
        payoff_proxy = np.maximum(x_em - K, 0)
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