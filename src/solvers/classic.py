import numpy as np
from ..dynamics import calculate_C

def price_monte_carlo(cfg, seed=42, return_raw=False):
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
        
        if return_raw:
            return payoff
        
        prices.append(np.mean(payoff))
        
    return prices

