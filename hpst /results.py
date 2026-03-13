"""Results storage and visualisation."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ExperimentResults:
    timestamp: str
    data_source: str
    n_points: int
    theorems_verified: int
    u_mean: float; u_std: float
    v_mean: float; v_std: float
    speed_mean: float; speed_std: float
    bernoulli_mean: float; bernoulli_range: float
    thresholds: Dict[str, float]
    adaptive_threshold: float
    pct_above_adaptive: float
    eigen_mean: float
    device: str


class ResultsSaver:
    def __init__(self, out_dir="results"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save_json(self, results: ExperimentResults, filename=None):
        if filename is None:
            filename = f"results_{results.timestamp}.json"
        path = os.path.join(self.out_dir, filename)
        with open(path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        print(f"✓ JSON saved: {path}")

    def plot_and_save(self, data, results: ExperimentResults):
        u = data['U'].value.cpu().numpy()
        v = data['V'].value.cpu().numpy()
        p = data['P'].value.cpu().numpy()
        speed = np.sqrt(u**2 + v**2)
        bernoulli = p + 0.5 * speed**2

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0,0].hist(u, bins=50, alpha=0.7)
        axes[0,0].set_title('U Distribution')

        axes[0,1].hist(v, bins=50, alpha=0.7, color='green')
        axes[0,1].set_title('V Distribution')

        axes[0,2].hist(speed, bins=50, alpha=0.7, color='red')
        axes[0,2].axvline(results.adaptive_threshold, color='k', linestyle='--',
                          label=f'Adaptive ({results.adaptive_threshold:.2f})')
        axes[0,2].legend()
        axes[0,2].set_title('Speed Distribution')

        axes[1,0].scatter(u, v, alpha=0.1, s=1)
        axes[1,0].set_title('U vs V')

        axes[1,1].hist(bernoulli, bins=50, alpha=0.7, color='purple')
        axes[1,1].set_title(f'Bernoulli (range={results.bernoulli_range:.3f})')

        thresholds = list(results.thresholds.keys())
        pcts = list(results.thresholds.values())
        axes[1,2].bar(range(len(thresholds)), pcts)
        axes[1,2].set_xticks(range(len(thresholds)))
        axes[1,2].set_xticklabels([f'v>{t}' for t in thresholds])
        axes[1,2].axhline(results.pct_above_adaptive, color='r', linestyle='--',
                          label=f'Adaptive: {results.pct_above_adaptive:.1f}%')
        axes[1,2].legend()
        axes[1,2].set_title('Points above threshold (%)')

        plt.suptitle(f'HPST Analysis – {results.data_source} data ({results.n_points} points)')
        plt.tight_layout()

        # Show (if in interactive environment) and save
        plt.show()
        plot_path = os.path.join(self.out_dir, f"plots_{results.timestamp}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {plot_path}")
        plt.close(fig)
