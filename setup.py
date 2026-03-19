from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="hpst-framework",
    version="1.0.0",
    author="Mohsen Mostafa",
    author_email="mohsen.mostafa.ai@outlook.com",
    description="Hybrid Physics-Spectral-Threshold Framework for Fluid Flow Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EPANG-Gen/HPST-Framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)from setuptools import setup, find_packages

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
