"""
NASS - Production Package Configuration
"""
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="nass-audio",
    version="1.0.0",
    author="nexuss0781",
    author_email="nexuss0781@gmail.com",
    description="Nexuss Audio Substrate System - High-performance AGI-grade audio processing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nexuss0781/NASS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "matplotlib>=3.4.0",
    ],
    entry_points={
        "console_scripts": [
            "nass=main:main",
        ],
    },
    keywords="audio processing STFT AGI machine learning pipeline parallel computing",
    license="MIT",
)
