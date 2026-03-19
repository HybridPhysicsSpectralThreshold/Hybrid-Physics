"""Core HPST algorithm implementation (52 lines)."""

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def adaptive_threshold(coords, u, v, n_clusters=5, alpha=0.7):
    """
    HPST adaptive thresholding with physics-informed region identification.
    
    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        Spatial coordinates (x, y)
    u, v : ndarray, shape (N,)
        Velocity components
    n_clusters : int
        Number of regions for clustering
    alpha : float
        Sensitivity parameter (threshold = μ + α·σ)
    
    Returns
    -------
    classification : ndarray, shape (N,)
        Binary mask (1 = above threshold)
    regions : ndarray, shape (N,)
        Region assignments
    thresholds : dict
        Threshold value for each region
    """
    # Step 1: Compute vorticity via finite differences
    nbrs = NearestNeighbors(n_neighbors=5).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    vort = np.zeros_like(u)
    for i in range(len(coords)):
        nb = idx[i, 1:]
        if len(nb) < 2: continue
        dx = coords[nb,0] - coords[i,0]
        dy = coords[nb,1] - coords[i,1]
        du = u[nb] - u[i]
        dv = v[nb] - v[i]
        A = np.stack([dx, dy], axis=1)
        try:
            du_dx, du_dy = np.linalg.lstsq(A, du, rcond=None)[0]
            dv_dx, dv_dy = np.linalg.lstsq(A, dv, rcond=None)[0]
            vort[i] = dv_dx - du_dy
        except:
            pass
    vort = np.nan_to_num(vort)
    
    # Step 2: Normalize features
    feat = np.stack([coords[:,0], coords[:,1], u, v, vort], axis=1)
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
    
    # Step 3: Spectral clustering
    clusters = SpectralClustering(n_clusters, affinity='rbf', 
                                  random_state=42).fit_predict(feat)
    
    # Step 4: Compute speed
    speed = np.sqrt(u**2 + v**2)
    classification = np.zeros_like(speed)
    thresholds = {}
    
    # Step 5: Adaptive threshold per region
    for r in range(n_clusters):
        in_idx = np.where(clusters == r)[0]
        out_idx = np.where(clusters != r)[0]
        
        if len(in_idx) == 0 or len(out_idx) == 0:
            T = speed.mean() + speed.std()
            classification[in_idx] = speed[in_idx] > T
            thresholds[r] = T
            continue
        
        # Distance-weighted statistics
        dist = cdist(coords[in_idx], coords[out_idx])
        w = 1.0 / (dist + 1e-8)
        w /= w.sum(axis=1, keepdims=True)
        
        mu_i = np.sum(w * speed[out_idx], axis=1)
        sigma_i = np.sqrt(np.sum(w * (speed[out_idx] - mu_i[:, None])**2, axis=1) + 1e-8)
        
        T_r = mu_i.mean() + alpha * sigma_i.mean()
        classification[in_idx] = speed[in_idx] > T_r
        thresholds[r] = T_r
    
    return classification, clusters, thresholds
