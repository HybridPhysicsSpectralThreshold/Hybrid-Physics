#!/usr/bin/env python3
"""Simple script to run the HPST experiment."""

from hpst.experiment import run_experiment

if __name__ == "__main__":
    run_experiment(prefer_real=True)
