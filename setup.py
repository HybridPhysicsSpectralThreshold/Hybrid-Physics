from setuptools import setup, find_packages

setup(
    name="hpst",
    version="0.1.0",
    author="Your Name",
    description="Hybrid Physics-Spectral-Threshold framework with theorem proving",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
    ],
    python_requires=">=3.7",
)
