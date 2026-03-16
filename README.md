# HPST: Hybrid Physics‑Spectral‑Threshold Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A unified framework integrating **symbolic theorem proving**, **physics‑informed constraints**, and **graph neural networks** for robust and interpretable fluid flow analysis.

## Installation

```python
git clone https://github.com/HybridPhysicsSpectralThreshold/Hybrid-Physics.git
cd hpst
pip install -r requirements.txt
```

Usage

Run the complete experiment:
```python
from hpst.experiment import run_experiment
run_experiment(prefer_real=True)   # tries real data, falls back to synthetic
```
📄 Citation

If you use this code in your research, please cite our paper:

```python
@article{mostafa2025hpst,
  title={HPST: A Hybrid Physics-Spectral-Threshold Framework for Fluid Flow Analysis with Theorem Proving and Graph Neural Networks},
  author={Mostafa, Mohsen},
  journal={Journal of Computational Physics (Under Review)},
  year={2025}
}
```

🔬 Overview

HPST combines three complementary perspectives into a unified pipeline:
Component	Description
🧮 Symbolic Theorem Proving	AC‑matching rewriting engine that verifies algebraic identities (commutativity, associativity, distributivity, transpose properties)
📊 Physical Analysis	Computes conservation laws (Bernoulli invariant), adaptive thresholds (μ+σ), and eigenvalue‑based flow characterization
🧠 Graph Neural Networks	EdgeConv‑based surrogate that learns velocity fields from scattered point clouds with physics‑informed divergence constraints

Key Achievement: Achieves up to R² = 0.208 on cylinder wake prediction while maintaining divergence error as low as 0.27 – a measure of physical consistency.
✨ Features
Symbolic Mathematics

Complete expression system for tensor operations (Add, Mul, MatMul, Transpose, Divergence, Vorticity, EigenDecomp, Threshold)

AC‑matching rewriting engine for automated theorem verification

FormalSystem container for managing axioms and theorems

Pre‑verified theorems: transpose of product, distributivity, associativity

Data Processing

Real CFD support: Loads .mat files from PINNs repository or CSV files with columns (x, y, u, v, p)

Synthetic generator: Realistic vortex‑street wake model (40,000 points by default)

Automatic fallback if real data unavailable

Coordinate normalization for stable training

Physical Analysis

Flow statistics (mean, std) for u, v, and speed

Bernoulli invariant computation

Fixed and adaptive (μ+σ) threshold analysis

Approximate eigenvalue magnitudes from velocity gradient matrices

Visualisation: distributions, scatter plots, threshold percentages

Graph Neural Network

Architecture: 5‑layer EdgeConv with 256 hidden units

Graph construction: k‑NN (k=10) with edge features = relative displacements

Physics‑informed loss: MSE + λ·𝔼[|∇·u|] with optimal λ = 0.1

Training: Adam (LR=5e-4) with ReduceLROnPlateau scheduler, 500 epochs

Memory efficient: Subsampling and OOM protection for 16GB GPUs

Evaluation & Visualisation

Automatic 80/20 train/test split

Baseline comparisons: k‑NN (k=5,20), linear regression, mean predictor

Spatial error maps: Identify regions of high prediction error

Divergence verification: Autograd‑based computation on regular grid

Multi‑run validation: Statistics over multiple initializations

Output & Reproducibility

JSON exports of all metrics and hyperparameters

LaTeX tables ready for publication

High‑resolution PNG plots

Complete session archiving with timestamps
