"""
眠気推定システム セットアップスクリプト
Drowsiness Estimation System Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drowsiness-estimation-system",
    version="1.0.0",
    author="Drowsiness Estimation Team",
    description="瞬き係数とLSTMを用いた眠気推定システム",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
    ],
    entry_points={
        'console_scripts': [
            'drowsiness-collect=src.drowsiness_data_collector:main',
            'drowsiness-train=src.train_drowsiness_model:main',
            'drowsiness-estimate=src.realtime_drowsiness_estimator:main',
        ],
    },
)
