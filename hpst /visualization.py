"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np

def plot_r2_comparison(results, save_path=None):
    """Plot R² comparison with error bars."""
    names = list(results.keys())
    r2_means = [results[n]['r2_mean'] for n in names]
    r2_stds = [results[n]['r2_std'] for n in names]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    plt.bar(x, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
    plt.xlabel('Experiment')
    plt.ylabel('R² Score')
    plt.title('R² Comparison with 95% Confidence')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_regions(coords, regions, thresholds, save_path=None):
    """Visualize region assignments and thresholds."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:,0], coords[:,1], c=regions, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Region ID')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Region Assignments (Thresholds: {thresholds})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_paper_figure(results, save_path='paper/figure1.png'):
    """Generate publication-ready figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # R² comparison
    names = list(results.keys())
    r2_means = [results[n]['r2_mean'] for n in names]
    r2_stds = [results[n]['r2_std'] for n in names]
    
    axes[0,0].bar(range(len(names)), r2_means, yerr=r2_stds, capsize=3, alpha=0.7)
    axes[0,0].set_xticks(range(len(names)))
    axes[0,0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0,0].set_ylabel('R²')
    axes[0,0].set_title('R² Comparison')
    axes[0,0].grid(True, alpha=0.3)
    
    # Best model scatter
    best_idx = np.argmax(r2_means)
    best_name = names[best_idx]
    best_data = results[best_name]['all_results'][0]
    
    axes[0,2].scatter(best_data['true'][:,0], best_data['pred'][:,0], alpha=0.3, s=2)
    axes[0,2].plot([best_data['true'][:,0].min(), best_data['true'][:,0].max()],
                   [best_data['true'][:,0].min(), best_data['true'][:,0].max()], 'r--')
    axes[0,2].set_xlabel('True U')
    axes[0,2].set_ylabel('Predicted U')
    axes[0,2].set_title(f'Best: {best_name}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
