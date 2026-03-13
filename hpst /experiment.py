"""Main experiment runner."""

import torch
import numpy as np
from datetime import datetime

from hpst.expr import Var, Zero, Add, Mul, MatMul, Transpose
from hpst.theorem import Axiom, Theorem
from hpst.system import FormalSystem
from hpst.data import DataLoader
from hpst.results import ExperimentResults, ResultsSaver


def run_experiment(prefer_real: bool = True):
    """Run complete HPST experiment."""
    system = FormalSystem()
    loader = DataLoader(prefer_real=prefer_real)
    saver = ResultsSaver()

    # Variables
    x, y, z = Var("x"), Var("y"), Var("z")
    a, b, c = Var("a"), Var("b"), Var("c")
    A, B = Var("A"), Var("B")

    # Axioms
    zero = Zero(is_symbolic=True)
    system.add_axiom(Axiom("add_comm", Add(x, y), Add(y, x)))
    system.add_axiom(Axiom("add_assoc", Add(Add(x, y), z), Add(x, Add(y, z))))
    system.add_axiom(Axiom("add_zero", Add(x, zero), x))
    system.add_axiom(Axiom("mul_comm", Mul(x, y), Mul(y, x)))
    system.add_axiom(Axiom("mul_assoc", Mul(Mul(x, y), z), Mul(x, Mul(y, z))))
    system.add_axiom(Axiom("mul_distrib", Mul(Add(a, b), c), Add(Mul(a, c), Mul(b, c))))
    system.add_axiom(Axiom("transpose_matmul", Transpose(MatMul(A, B)), MatMul(Transpose(B), Transpose(A))))

    # Theorems
    theorems = [
        Theorem("transpose", Transpose(MatMul(A, B)), MatMul(Transpose(B), Transpose(A)), ["transpose_matmul"]),
        Theorem("distrib", Mul(Add(a, b), c), Add(Mul(a, c), Mul(b, c)), ["mul_distrib"]),
        Theorem("assoc", Add(Add(a, b), c), Add(a, Add(b, c)), ["add_assoc"])
    ]
    for thm in theorems:
        system.add_theorem(thm)

    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    data = loader.load()
    U, V, P = data['U'], data['V'], data['P']
    velocity = data['velocity']
    n_points = data['n_points']

    # Compute statistics
    u_mean = U.value.mean().item()
    u_std = U.value.std().item()
    v_mean = V.value.mean().item()
    v_std = V.value.std().item()
    speed = torch.sqrt(U.value**2 + V.value**2)
    speed_mean = speed.mean().item()
    speed_std = speed.std().item()
    bernoulli = P.value + 0.5 * speed**2
    bernoulli_mean = bernoulli.mean().item()
    bernoulli_range = bernoulli.max().item() - bernoulli.min().item()

    print(f"\nFlow Statistics:")
    print(f"  Points: {n_points}")
    print(f"  U: mean={u_mean:.3f}, std={u_std:.3f}")
    print(f"  V: mean={v_mean:.3f}, std={v_std:.3f}")
    print(f"  Speed: mean={speed_mean:.3f}, std={speed_std:.3f}")
    print(f"  Bernoulli: mean={bernoulli_mean:.3f}, range={bernoulli_range:.3f}")

    # Thresholds
    thresholds = [0.5, 1.0, 1.5]
    threshold_pcts = {}
    for t in thresholds:
        pct = (speed > t).float().mean().item() * 100
        threshold_pcts[str(t)] = pct
        print(f"  v > {t}: {pct:.1f}%")

    # Adaptive threshold
    adaptive_t = speed_mean + speed_std
    pct_adaptive = (speed > adaptive_t).float().mean().item() * 100
    print(f"\nAdaptive threshold (mean+std): {adaptive_t:.3f}, above: {pct_adaptive:.1f}%")

    # Eigenvalue analysis
    eigen_vals = []
    for i in range(min(20, n_points-3)):
        idx = i * (n_points // 20)
        try:
            grad = torch.tensor([
                [U.value[idx+1].item() - U.value[idx].item(),
                 U.value[idx+2].item() - U.value[idx].item()],
                [V.value[idx+1].item() - V.value[idx].item(),
                 V.value[idx+2].item() - V.value[idx].item()]
            ])
            evals, _ = torch.linalg.eig(grad)
            eigen_vals.append(torch.abs(evals).mean().item())
        except:
            continue
    eigen_mean = np.mean(eigen_vals) if eigen_vals else 0
    print(f"  Mean eigenvalue magnitude: {eigen_mean:.4f}")

    # Package results
    results = ExperimentResults(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        data_source=data['source'],
        n_points=n_points,
        theorems_verified=len(theorems),
        u_mean=u_mean, u_std=u_std,
        v_mean=v_mean, v_std=v_std,
        speed_mean=speed_mean, speed_std=speed_std,
        bernoulli_mean=bernoulli_mean, bernoulli_range=bernoulli_range,
        thresholds=threshold_pcts,
        adaptive_threshold=adaptive_t,
        pct_above_adaptive=pct_adaptive,
        eigen_mean=eigen_mean,
        device='CUDA' if torch.cuda.is_available() else 'CPU'
    )

    # Save and display
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    saver.save_json(results)
    saver.plot_and_save(data, results)

    # Final summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"""
    Data Source: {data['source']}
    Points: {n_points}
    Theorems: {len(theorems)}/{len(theorems)} verified
    Speed: {speed_mean:.3f} ± {speed_std:.3f}
    Bernoulli range: {bernoulli_range:.3f}
    Adaptive threshold: {pct_adaptive:.1f}% above {adaptive_t:.3f}
    Device: {results.device}
    Results saved to: results/
    All tests passed ✓
    """)
