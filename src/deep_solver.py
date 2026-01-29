import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from .dynamics import calculate_C

# --- 1. 神经网络结构 (The Brain) ---
class Net(nn.Module):
    def __init__(self,cfg, target_strike):
        super().__init__()
        # Y_0: The Price we want
        s0 = cfg['x0'] 
        
        # Smart Initialization
        # max(S0 - K, 0)
        intrinsic_val = max(s0 - target_strike, 0.0) 
        
        # 初始猜测 = 内在价值 + 2.0 (防止梯度消失)
        start_guess = intrinsic_val + 2.0
        
        self.y_init = nn.Parameter(torch.tensor(float(start_guess)))
        
        # Hedge Strategy
        # Input Dimension 3: (t, x, sigma)
        # Output Dimension 2: (Z1, Z2) 对应两个布朗运动
        self.z_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64), # BatchNorm 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)    # Output: Z1, Z2
        )
        
    def forward(self, x):
        return self.z_net(x)


def train_deep_bsde(cfg, target_strike):
    """
    Trains the Deep BSDE solver for a specific strike.
    Returns:
        learned_price (float): The final Y0.
        loss_history (list): Loss values for visualization.
        price_history (list): Y0 convergence history. 
        final_z_path (list): The hedging path of the last epoch. 
        training_time (float): Total seconds used.
    """
    device = cfg["device"]
    print(f"[Deep BSDE] Start Training for Strike K={target_strike}...")
    
    # 打印具体的硬件信息
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"    Hardware: GPU ({gpu_name}) is ON.")
    elif device.type == 'mps': # 适配 Mac M1/M2/M3
        print(f"    Hardware: Apple Silicon (MPS) is ON.")
    else:
        print(f"    Hardware: CPU (Warning: Training might be slow)")
        
    dt = cfg["dt"]
    n_steps = cfg["n_steps"]
    
    # Prepare constants
    sqrt_dt = torch.tensor(np.sqrt(dt), device=device).float()
    
    # Initializing the model
    model = Net(cfg, target_strike).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    #容器
    loss_history = []
    price_history = []   # 用于记录每一轮 Y0 的变化 (画中间的图)
    final_z_path = []    # 用于记录最后一次模拟的完整路径 (画右边的图)
    
    epochs = cfg.get('n_epochs', 30)#  Here is the Epoch
    batch_size = cfg['batch_size']# Here is the batch size
    
    print(f" [Deep BSDE] Start Training for Strike K={target_strike}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- A.  Batch Simulation ---
        
        # t=0 initialize
        x = torch.full((batch_size, 1), cfg["x0"], device=device)
        sigma = torch.full((batch_size, 1), cfg["sigma0"], device=device)
        y = model.y_init.expand(batch_size, 1) # Y0 是我们要学的
        
        # Current container
        current_path_z = []
        
        # Iterate in times
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((batch_size, 1), t_val, device=device)
            
            # 1. Using NN to dicide the Hedge Strategy Z
            state = torch.cat([t, x, sigma], dim=1)
            z = model(state) 
            # 我们只需要 t=0 (也就是 i=0) 时的 Z
            if i == 0:
                # z[:, 0] 是对股票价格的对冲 (Delta)
                # z[:, 1] 是对波动率的对冲 (Vega风险)
                # 我们取第一列，求平均，并断开梯度(.detach)
                model.saved_z_init = z[:, 0].mean().detach()
                
            # Only Record in the final epoch
            if epoch == epochs - 1:
                # 提取数据 (Batch 0, Dimension 0 for Delta)
                z_val = z[0, 0].detach().cpu().item()
                x_val = x[0, 0].detach().cpu().item()
                
                # 归一化 Delta = Z / (sigma * X)
                # 加 1e-9 防止除以 0
                delta_val = z_val / (cfg["sigma0"] * x_val + 1e-9)
                current_path_z.append(delta_val)
            
            # 2. Generate the noize
            dw1 = torch.randn_like(x) * sqrt_dt
            dw2 = torch.randn_like(x) * sqrt_dt
            
            # 3. The FSDE
            c_val = calculate_C(x)
            x_next = x + sigma * c_val * dw1
            sigma_next = sigma + cfg["epsilon"] * sigma * (
                cfg["rho"] * dw1 + torch.sqrt(1 - torch.tensor(cfg["rho"]**2, device=device)) * dw2
            )
            
            # 4. the BSDE
            # dY = Z * dW ( r=0)
            y_next = y + z[:, 0:1] * dw1 + z[:, 1:2] * dw2
            
            # Updating
            x = x_next
            sigma = sigma_next
            y = y_next
            
        # --- B. Calculate The Loss ---
        # Target: Y_T == Payoff(X_T)
        payoff = torch.maximum(x - target_strike, torch.tensor(0.0, device=device))
        
        # Loss = MSE
        loss = torch.mean((y - payoff)**2)
        
        # --- C. manchine is learning... ---
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        price_history.append(model.y_init.item()) # 记录这一轮算出的 Y0
        
        if epoch == epochs - 1:
            final_z_path = current_path_z # 保存最后一轮的路径
        
        # Printing Progress
        log_interval = max(1, epochs // 5)
        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:<4} | Loss: {loss.item():.6f} | Price: {model.y_init.item():.4f}")

    total_time = time.time() - start_time
    return model.y_init.item(), loss_history, price_history, final_z_path, total_time