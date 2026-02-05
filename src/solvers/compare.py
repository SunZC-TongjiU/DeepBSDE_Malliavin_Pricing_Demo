import numpy as np
from ..dynamics import calculate_C, calculate_dC_dx, calculate_V_numpy

def price_monte_carlo(cfg, seed=42, return_raw=False):
    """
    Computes prices using Euler-Maruyama simulation.
    Ground Truth benchmark.
    """
    T= cfg["T"]
    n_paths, n_steps = cfg["n_paths"], cfg["n_steps"]
    dt = T/n_steps
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
        # payoff = np.maximum(x_mc - K,0)
        payoff = x_mc**2
        
        if return_raw:
            return payoff
        
        prices.append(np.mean(payoff))
        
    return prices

def price_malliavin(cfg, seed=42, return_raw=False):
    """
    Computes prices using 1st-order Malliavin Asymptotic Expansion.
    """
    T, dt = cfg["T"], cfg["dt"]
    n_paths, n_steps = cfg["n_paths"], cfg["n_steps"]
    dt = T/n_steps
    x0, sigma0 = cfg["x0"], cfg["sigma0"]
    epsilon, rho = cfg["epsilon"], cfg["rho"]
    rho_bar = np.sqrt(1-rho*rho)
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
        
        x_next = x_em + sigma_em * c_val * dw1
        sigma_next = sigma_em + epsilon * sigma_em * (rho * dw1 + np.sqrt(1 - rho**2) * dw2)
        
        
        # Diffusion Matrix
        V = calculate_V_numpy(x_em,sigma_em,epsilon, rho, rho_bar)
        V_T = np.swapaxes(V, -1, -2)
        V_11 = V[:,0,0]
        V_21 = V[:,1,0]
        
        # Calculate Malliavin Covariance Matrix
        A= V @ V_T
        A_inv = np.linalg.inv(A)
        Temp= A_inv @ V
        H_111 = Temp[:,0,0]**3
        
        #Compute Malliavin Weights
        H_111 = H_111* dw1*(dw1**2-3*dt)/(2*dt)*(V_11)**2 * (V_11 *dc_val* sigma_em +c_val* V_21)
        
        L1V11 = (sigma_em * c_val) * (sigma_em * dc_val) + (epsilon * rho * sigma_em) * c_val 
        L2V11 = (epsilon * rho_bar * sigma_em) * c_val
        H_11 = (1.0 / (4.0 * V_11**2)) * (L1V11**2 + L2V11**2) * (dw1**2 - dt)
        # Weight Formula
        weights = weights * (1.0 + H_111 +H_11)
        
        #Update Data
        x_em = x_next
        sigma_em = sigma_next
    
    # 4. Pricing
    prices = []
    for K in cfg["strikes"]:
        # payoff = np.maximum(x_em - K,0)
        payoff = x_em**2
        payoff = payoff*weights
        
        if return_raw:
            return payoff
        
        prices.append(np.mean(payoff))
        
    return prices
