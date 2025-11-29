"""
Setup script for Esercitazione 1 package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="esercitazione_1_mri_noise",
    version="1.0.0",
    author="Bioimmagini Positano",
    description="MRI Noise Analysis Tools - Python conversion from MATLAB exercises",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimmagini/positano",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pydicom>=2.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "scikit-image>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-calcolo-sd=src.calcolo_sd:main",
            "run-esempio-calcolo-sd=src.esempio_calcolo_sd:main",
            "run-test-m-sd=src.test_m_sd:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
        "data": ["*.dcm"],
    },
    zip_safe=False,
)
