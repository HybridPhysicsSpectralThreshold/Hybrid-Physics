"""Data loaders for all flow configurations."""

import numpy as np
import torch

def load_synthetic_data(reynolds=100, n_points=10000):
    """Generate synthetic vortex street data."""
    print(f"  Generating Re={reynolds} data with {n_points} points...")
    nx = int(np.sqrt(n_points * 2))
    ny = n_points // nx
    x = np.linspace(-2, 10, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_points], Y.flatten()[:n_points]

    params = {100: (0.5, 0.5), 1000: (0.8, 0.3), 3900: (1.2, 0.2)}
    gamma, a = params.get(reynolds, (0.5, 0.5))
    
    U_inf, x0 = 1.0, 2.0
    u = U_inf * np.ones_like(X)
    v = np.zeros_like(X)
    
    for i in range(20):
        sign = (-1) ** i
        xv, yv = x0 + i * 1.5, 0.5 * sign
        dx, dy = X - xv, Y - yv
        r2 = dx**2 + dy**2 + a**2
        u += -sign * gamma * dy / r2 * (1 - np.exp(-r2 / a**2))
        v += sign * gamma * dx / r2 * (1 - np.exp(-r2 / a**2))
    
    u *= (1 - 0.3 * np.exp(-0.3 * (X - x0)) * (X > x0))
    u, v = np.nan_to_num(u, nan=U_inf), np.nan_to_num(v, nan=0.0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return {
        'coords': torch.tensor(np.stack([X, Y], axis=1), dtype=torch.float32, device=device),
        'u': torch.tensor(u, dtype=torch.float32, device=device),
        'v': torch.tensor(v, dtype=torch.float32, device=device),
        'n_points': len(u),
        'source': f'synthetic_Re{reynolds}'
    }

def load_airfoil_data(n_points=10000):
    """Generate airfoil flow data."""
    print("  Generating airfoil flow data...")
    nx = int(np.sqrt(n_points * 2))
    ny = n_points // nx
    x = np.linspace(-2, 6, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_points], Y.flatten()[:n_points]
    
    U_inf = 1.0
    u = U_inf * np.ones_like(X)
    v = 0.2 * np.sin(2*np.pi*X/3) * np.exp(-0.5*(Y-0.5)**2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return {
        'coords': torch.tensor(np.stack([X, Y], axis=1), dtype=torch.float32, device=device),
        'u': torch.tensor(u, dtype=torch.float32, device=device),
        'v': torch.tensor(v, dtype=torch.float32, device=device),
        'n_points': len(u),
        'source': 'airfoil'
    }

def load_bfs_data(n_points=10000):
    """Generate backward-facing step flow."""
    print("  Generating BFS flow data...")
    nx = int(np.sqrt(n_points * 2))
    ny = n_points // nx
    x = np.linspace(0, 10, nx)
    y = np.linspace(-1, 2, ny)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_points], Y.flatten()[:n_points]
    
    U_inf = 1.0
    u = U_inf * np.ones_like(X)
    v = np.zeros_like(X)
    
    recirc = (X > 2) & (X < 5) & (Y < 0.5)
    u[recirc] = 0.3 * U_inf
    v[recirc] = -0.1 * U_inf * np.sin(2*np.pi*(X[recirc]-2)/3)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return {
        'coords': torch.tensor(np.stack([X, Y], axis=1), dtype=torch.float32, device=device),
        'u': torch.tensor(u, dtype=torch.float32, device=device),
        'v': torch.tensor(v, dtype=torch.float32, device=device),
        'n_points': len(u),
        'source': 'bfs'
    }

def load_real_piv_data():
    """Load synthetic PIV data with noise."""
    data = load_synthetic_data(reynolds=100, n_points=5000)
    noise_level = 0.05
    data['u'] = data['u'] + torch.randn_like(data['u']) * noise_level
    data['v'] = data['v'] + torch.randn_like(data['v']) * noise_level
    data['source'] = 'real_piv'
    return data
