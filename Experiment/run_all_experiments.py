#!/usr/bin/env python
"""Run all HPST framework experiments."""

import argparse
import yaml
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import hpst
from hpst.utils import set_seed, convert_to_serializable

def load_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(config, n_seeds=10, epochs=500):
    """Run a single experiment with multiple seeds."""
    print(f"\n{'='*60}")
    print(f"{config['name']} - {n_seeds} seeds")
    print(f"{'='*60}")
    
    all_results = []
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")
        set_seed(42 + seed)
        
        # Load data
        if config['data_source'] == 're100':
            data = hpst.data.load_synthetic_data(reynolds=100, n_points=10000)
        elif config['data_source'] == 're1000':
            data = hpst.data.load_synthetic_data(reynolds=1000, n_points=10000)
        elif config['data_source'] == 're3900':
            data = hpst.data.load_synthetic_data(reynolds=3900, n_points=10000)
        elif config['data_source'] == 'airfoil':
            data = hpst.data.load_airfoil_data()
        elif config['data_source'] == 'bfs':
            data = hpst.data.load_bfs_data()
        elif config['data_source'] == 'real':
            data = hpst.data.load_real_piv_data()
        
        coords = data['coords'].cpu().numpy()
        u = data['u'].cpu().numpy()
        v = data['v'].cpu().numpy()
        
        # Split data
        idx = np.random.permutation(len(coords))
        train_idx = idx[:int(0.72*len(coords))]
        val_idx = idx[int(0.72*len(coords)):int(0.9*len(coords))]
        test_idx = idx[int(0.9*len(coords)):]
        
        X_train, X_val, X_test = coords[train_idx], coords[val_idx], coords[test_idx]
        u_train, u_val, u_test = u[train_idx], u[val_idx], u[test_idx]
        v_train, v_val, v_test = v[train_idx], v[val_idx], v[test_idx]
        
        # Normalize
        u_mean, u_std = u_train.mean(), u_train.std() + 1e-8
        v_mean, v_std = v_train.mean(), v_train.std() + 1e-8
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        
        # Create model
        if config['model_type'] == 'mlp':
            model = hpst.models.MLP().to(device)
        else:
            model = hpst.models.GraphTransformer().to(device)
        
        # Train
        trainer = hpst.Trainer(model)
        history = trainer.train(
            X_train_t, u_train, v_train, u_mean, u_std,
            X_val_t, u_val, v_val, v_mean, v_std,
            epochs=epochs
        )
        
        # Evaluate
        metrics = trainer.evaluate(X_test_t, u_test, v_test, u_mean, u_std, v_mean, v_std)
        all_results.append(metrics)
        
        print(f"  Seed {seed+1} R²: {metrics['r2']:.4f}")
    
    # Aggregate
    aggregated = {
        'name': config['name'],
        'n_seeds': n_seeds,
        'r2_mean': float(np.mean([r['r2'] for r in all_results])),
        'r2_std': float(np.std([r['r2'] for r in all_results])),
        'r2_u_mean': float(np.mean([r['r2_u'] for r in all_results])),
        'r2_u_std': float(np.std([r['r2_u'] for r in all_results])),
        'r2_v_mean': float(np.mean([r['r2_v'] for r in all_results])),
        'r2_v_std': float(np.std([r['r2_v'] for r in all_results])),
        'all_results': convert_to_serializable(all_results)
    }
    
    return aggregated

def main():
    parser = argparse.ArgumentParser(description='Run HPST experiments')
    parser.add_argument('--n_seeds', type=int, default=10, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='experiments/results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define experiments
    experiments = [
        {'name': 'MLP-Baseline', 'model_type': 'mlp', 'data_source': 're100'},
        {'name': 'GNN-Re100', 'model_type': 'gnn', 'data_source': 're100'},
        {'name': 'GNN-Re1000', 'model_type': 'gnn', 'data_source': 're1000'},
        {'name': 'GNN-Re3900', 'model_type': 'gnn', 'data_source': 're3900'},
        {'name': 'GNN-Airfoil', 'model_type': 'gnn', 'data_source': 'airfoil'},
        {'name': 'GNN-BFS', 'model_type': 'gnn', 'data_source': 'bfs'},
        {'name': 'HPST-Re100', 'model_type': 'gnn', 'data_source': 're100'},
        {'name': 'HPST-Re1000', 'model_type': 'gnn', 'data_source': 're1000'},
        {'name': 'HPST-Re3900', 'model_type': 'gnn', 'data_source': 're3900'},
        {'name': 'HPST-Airfoil', 'model_type': 'gnn', 'data_source': 'airfoil'},
        {'name': 'HPST-BFS', 'model_type': 'gnn', 'data_source': 'bfs'},
        {'name': 'Real-PIV', 'model_type': 'gnn', 'data_source': 'real'},
    ]
    
    all_results = {}
    for exp in experiments:
        all_results[exp['name']] = run_experiment(exp, args.n_seeds, args.epochs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"all_results_{timestamp}.json")
    with open(out_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    # Print summary
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY")
    print("="*100)
    print(f"{'Experiment':<20} {'R² (mean±std)':<20}")
    print("-"*50)
    for name, res in all_results.items():
        print(f"{name:<20} {res['r2_mean']:.4f}±{res['r2_std']:.4f}")

if __name__ == "__main__":
    main()
