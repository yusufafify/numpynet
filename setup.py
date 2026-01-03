"""Setup script for NumPyNet package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="numpynet",
    version="0.1.0",
    author="NumPyNet Contributors",
    author_email="",
    description="A lightweight, educational deep learning framework built from scratch using only NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yusufafify/numpynet",  # Update with actual repo URL
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "examples": ["matplotlib>=3.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords="deep-learning neural-network numpy education machine-learning",
    project_urls={
        "Documentation": "https://github.com/yusufafify/numpynet#readme",
        "Source": "https://github.com/yusufafify/numpynet",
        "Bug Reports": "https://github.com/yusufafify/numpynet/issues",
    },
)
