"""Data loader for real/synthetic cylinder wake data."""

import os
import urllib.request
import ssl
import numpy as np
import scipy.io
import torch
from hpst.expr import Const


class DataLoader:
    def __init__(self, prefer_real: bool = True):
        self.prefer_real = prefer_real
        self.source = "unknown"

    def load(self):
        if self.prefer_real:
            try:
                return self._load_real()
            except Exception as e:
                print(f"Real data failed: {e}")
                print("Falling back to synthetic...")
                return self._load_synthetic()
        else:
            return self._load_synthetic()

    def _load_real(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        url = "https://github.com/maziarraissi/PINNs/raw/master/main/Data/cylinder_nektar_wake.mat"
        filename = "cylinder_nektar_wake.mat"
        if not os.path.exists(filename):
            print("Downloading real CFD data...")
            urllib.request.urlretrieve(url, filename)
        data = scipy.io.loadmat(filename)
        # Try different key names
        keys = list(data.keys())
        if 'U_star' in data and 'V_star' in data and 'p_star' in data:
            u = data['U_star'][:, 0]
            v = data['V_star'][:, 0]
            p = data['p_star'][:, 0]
        elif 'u' in data and 'v' in data and 'p' in data:
            u = data['u'].flatten()
            v = data['v'].flatten()
            p = data['p'].flatten()
        else:
            raise KeyError(f"Unknown variable names in {keys}")
        self.source = "real"
        print(f"✓ Loaded real data: {len(u)} points")
        return self._to_tensors(u, v, p)

    def _load_synthetic(self):
        print("Generating synthetic data...")
        n = 5000
        x = np.linspace(-2, 10, n)
        u = 1.0 + 0.1 * np.sin(0.5 * x) * np.exp(-0.1 * x)
        v = -0.2 + 0.05 * np.cos(0.5 * x) * np.exp(-0.1 * x)
        p = -0.5 * (u**2 + v**2)
        self.source = "synthetic"
        print(f"✓ Generated synthetic data: {n} points")
        return self._to_tensors(u, v, p)

    def _to_tensors(self, u, v, p):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        u_t = torch.tensor(u, dtype=torch.float32, device=device)
        v_t = torch.tensor(v, dtype=torch.float32, device=device)
        p_t = torch.tensor(p, dtype=torch.float32, device=device)
        return {
            'U': Const(u_t),
            'V': Const(v_t),
            'P': Const(p_t),
            'velocity': Const(torch.stack([u_t, v_t], dim=-1)),
            'n_points': len(u),
            'source': self.source
        }
