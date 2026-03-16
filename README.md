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
