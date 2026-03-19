"""
HPST Framework: Hybrid Physics-Spectral-Threshold for Fluid Flow Analysis

This package provides tools for physics-informed adaptive thresholding
and graph neural network-based velocity field prediction.
"""

__version__ = "1.0.0"
__author__ = "Mohsen Mostafa"

from . import core
from . import threshold
from . import models
from . import data
from . import utils
from . import visualization

__all__ = [
    'core',
    'threshold',
    'models',
    'data',
    'utils',
    'visualization',
]
