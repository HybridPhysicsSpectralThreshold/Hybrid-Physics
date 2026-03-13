# HPST: Hybrid Physics‑Spectral‑Threshold Framework

A lightweight theorem‑proving framework for tensor mathematics with physical data validation.  
Combines symbolic algebra (axioms, theorems, AC‑matching) with real/synthetic fluid dynamics data to compute conservation laws, adaptive thresholds, and eigenvalue‑based flow characterisation.

## Features

- Symbolic expression system for tensor operations (Add, Mul, MatMul, Transpose, Divergence, Vorticity, EigenDecomp, Threshold)
- AC‑matching rewriting engine to verify algebraic identities
- FormalSystem container for axioms and theorems
- Data loader that fetches real cylinder wake data (with fallback to synthetic)
- Physical analysis: flow statistics, Bernoulli invariant, threshold analysis, adaptive threshold (mean+std), eigenvalue magnitudes
- Results saved as JSON + PNG plots
- Fully reproducible; runs on Kaggle/Colab with a single click

## Installation

```python
git clone https://github.com/yourusername/hpst.git
cd hpst
pip install -r requirements.txt
```

Usage

Run the complete experiment:
```python
from hpst.experiment import run_experiment
run_experiment(prefer_real=True)   # tries real data, falls back to synthetic
```
Citation
```python
@article{hpst2025,
  title={HPST: Hybrid Physics‑Spectral‑Threshold framework for theorem proving and fluid data analysis},
  author={Your Name},
  journal={Journal of Computational Physics},
  year={2025}
}
```
