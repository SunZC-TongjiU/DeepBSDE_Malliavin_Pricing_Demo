import numpy as np
from ..dynamics import calculate_C, calculate_dC_dx, calculate_V_numpy,calculate_d2C_dx2

# --- Method 1: MC with higher-order shceme, using Malliavin IBP
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
        dw1 = z1 * sqrt_dt
        dw2 = z2 * sqrt_dt
        
        c_val = calculate_C(x_em)
        dc_val = calculate_dC_dx(x_em)
        ddc_val = calculate_d2C_dx2(x_em)
        
        x_next = x_em + sigma_em * c_val * dw1
        sigma_next = sigma_em + epsilon * sigma_em * (rho * dw1 + np.sqrt(1 - rho**2) * dw2)
        
        
        # Diffusion Matrix
        V = calculate_V_numpy(x_em,sigma_em,epsilon, rho, rho_bar)
        V_T = np.swapaxes(V, -1, -2)
        V_11 = V[:,0,0]
        V_12 = V[:,0,1]
        V_21 = V[:,1,0]
        V_22 = V[:,1,1]
        
        partial_1_V11= sigma_em*dc_val
        partial_2_V11 = c_val
        partial_11_V11 = sigma_em*ddc_val
        partial_12_V11 = dc_val
        
        #Get Gradient of Diffusion Matrix
        L1V11 = V_11*partial_1_V11 + V_21*partial_2_V11
        L2V11 = V_22 *partial_2_V11
        L0V11 = 0.5*V_11*V_11*partial_11_V11 + V_11*V_21*partial_12_V11
        
        
        # # Calculate Malliavin Covariance Matrix
        # A= V @ V_T
        # # 在计算 A_inv 之前
        # det_A = np.linalg.det(A)
        # if np.min(det_A) < 1e-8:
        #     print(f"Warning: A is nearly singular! Min Det: {np.min(det_A)}")
        M = np.linalg.inv(V_T)
        
        #Generate Hermit Polynumial Tensor
            #3 dimension
            #calculate wick polynomial
        Q1 = dw1**2 - dt     #Quadratic
        Q2 = dw2**2 - dt
        C1 = dw1**3 - 3*dt*dw1    #Cubic
        C2 = dw2**3 - 3*dt*dw2
        
        P3 = np.zeros((n_paths, 2, 2, 2))
        P3[:, 0, 0, 0] = C1
        P3[:, 1, 1, 1] = C2
        
        val_001 = Q1 * dw2
        P3[:, 0, 0, 1] = val_001
        P3[:, 0, 1, 0] = val_001
        P3[:, 1, 0, 0] = val_001
        
        val_011 = dw1 * Q2
        P3[:, 0, 1, 1] = val_011
        P3[:, 1, 0, 1] = val_011
        P3[:, 1, 1, 0] = val_011
        
        # Einstein Summation Notation:
        # n: batch dimension
        # i, j, k: output indices (index of Malliavin Weight H: 1, 2)
        # a, b, c: summation indices (index of Brownian Motion W: 1, 2)
        H_tensor_3 = np.einsum('nia, njb, nkc, nabc -> nijk', M, M, M, P3) / dt**3
        
            #2 dimension
            #calculate wick polynomial
        Q1 = dw1**2 - dt     #Quadratic
        Q2 = dw2**2 - dt
        
        P2 = np.zeros((n_paths, 2, 2))
        P2[:, 0, 0] = Q1
        P2[:, 1, 1] = Q2
        
        val_01 = dw1 * dw2
        P2[:, 0, 1] = val_01
        P2[:, 1, 0] = val_01
        
        # Einstein Summation Notation:
        # n: batch dimension
        # i, j, k: output indices (index of Malliavin Weight H: 1, 2)
        # a, b, c: summation indices (index of Brownian Motion W: 1, 2)
        H_tensor_2 = np.einsum('nia, njb, nab -> nij', M, M, P2) / dt**2
        
        #Get Malliavin Weights
        H_111 = H_tensor_3[:, 0, 0, 0]
        H_121 = H_tensor_3[:, 0, 1, 0]
        H_221 = H_tensor_3[:, 1, 1, 0]
        H_11 = H_tensor_2[:,0,0]
        H_21 = H_tensor_2[:,1,0]
        
        
        # Weight Formula
        
        term1 =  (H_121 * V_11 * V_22 + H_221 * V_22 * V_21) * L2V11 * 0.5 * (dt**2)
            
        term2 =  (H_111 * (V_11**2) + 2 * H_121 * V_11 * V_21 + H_221 * (V_21**2))* L1V11 * 0.5 * (dt**2)
            
        term3 =  (H_11 * V_11 + H_21 * V_21) * L0V11 * 0.5 * (dt**2)
            
        term4 =  H_11 * ((L1V11)**2 + (L2V11)**2) * 0.25 * (dt**2)
            
        PI = term1 + term2 + term3 + term4
        weights = weights * (1.0 + PI)
        
        #Update Data
        x_em = x_next
        sigma_em = sigma_next
    
    # 4. Pricing
    prices = []
    for K in cfg["strikes"]:
        payoff = np.maximum(x_em - K,0)
        payoff = payoff*weights
        
        if return_raw:
            return payoff
        
        prices.append(np.mean(payoff))
        
    return prices
