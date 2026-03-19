"""Tests for core HPST algorithm."""

import numpy as np
import pytest
from hpst.core import adaptive_threshold

def test_adaptive_threshold_basic():
    """Test basic functionality."""
    # Create simple test data
    coords = np.random.randn(100, 2)
    u = np.random.randn(100)
    v = np.random.randn(100)
    
    classification, regions, thresholds = adaptive_threshold(coords, u, v)
    
    assert classification.shape == (100,)
    assert regions.shape == (100,)
    assert len(thresholds) == 5
    assert classification.dtype == bool

def test_adaptive_threshold_edge_cases():
    """Test edge cases."""
    coords = np.zeros((10, 2))
    u = np.ones(10)
    v = np.zeros(10)
    
    classification, regions, thresholds = adaptive_threshold(coords, u, v, n_clusters=2)
    assert np.all(classification >= 0)
    assert np.all(classification <= 1)
