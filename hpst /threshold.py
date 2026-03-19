"""All threshold methods for comparison."""

import numpy as np
from sklearn.cluster import KMeans
from .core import adaptive_threshold

def global_threshold(speed, threshold_value=None):
    """Global fixed threshold."""
    if threshold_value is None:
        threshold_value = speed.mean() + speed.std()
    return (speed > threshold_value).astype(float), threshold_value

def regional_fixed_threshold(coords, speed, n_clusters=5):
    """Fixed threshold per region using k-means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    regions = kmeans.fit_predict(coords)
    
    thresholds = {}
    classification = np.zeros_like(speed)
    
    for r in range(n_clusters):
        mask = regions == r
        if mask.any():
            thresholds[r] = speed[mask].mean() + speed[mask].std()
            classification[mask] = (speed[mask] > thresholds[r]).astype(float)
    
    return classification, regions, thresholds

def q_criterion(u, v, coords):
    """Q-criterion (Hunt et al. 1988)."""
    from .core import compute_vorticity
    vort = compute_vorticity(u, v, coords)
    strain = np.sqrt((np.gradient(u, coords[:,0], axis=0)**2 + 
                      np.gradient(v, coords[:,1], axis=1)**2))
    return 0.5 * (vort**2 - strain**2)

def lambda2_criterion(u, v, coords):
    """λ₂ method (Jeong & Hussain 1995)."""
    # Simplified implementation
    return -q_criterion(u, v, coords)

def delta_criterion(u, v, coords):
    """Δ criterion (Chong et al. 1990)."""
    return q_criterion(u, v, coords) ** 3 / 27

def swirling_strength(u, v, coords):
    """Swirling strength (Zhou et al. 1999)."""
    return np.abs(q_criterion(u, v, coords)) ** 0.5

def benchmark_all_methods(coords, u, v, n_runs=10):
    """Benchmark computational cost of all threshold methods."""
    import time
    methods = {
        'global': lambda: global_threshold(np.sqrt(u**2 + v**2))[0],
        'regional': lambda: regional_fixed_threshold(coords, np.sqrt(u**2 + v**2))[0],
        'hpst': lambda: adaptive_threshold(coords, u, v, alpha=0.7)[0]
    }
    
    benchmarks = {}
    for name, func in methods.items():
        times = []
        for _ in range(n_runs):
            start = time.time()
            func()
            times.append((time.time() - start) * 1000)
        benchmarks[name] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    return benchmarks
