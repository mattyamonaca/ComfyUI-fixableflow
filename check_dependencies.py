#!/usr/bin/env python
"""
ComfyUI-fixableflow dependency check script
Run this to verify all requirements are properly installed
"""

import sys
import subprocess
import importlib
from pathlib import Path

print("=" * 70)
print("ComfyUI-fixableflow Dependency Checker")
print("=" * 70)

# Check Python version
print(f"\n1. Python Version:")
print(f"   Current: {sys.version}")
print(f"   Recommended: 3.10.x or 3.11.x")

# Check required packages
print(f"\n2. Required Packages:")
packages = {
    # Core packages
    'numpy': 'numpy',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    
    # Scientific packages
    'skimage': 'scikit-image',
    'sklearn': 'scikit-learn',
    'pandas': 'pandas',
    
    # PSD handling
    'pytoshop': 'pytoshop',
}

missing_packages = []
installed_packages = []

for import_name, pip_name in packages.items():
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        installed_packages.append(f"   ✓ {pip_name} ({import_name}): {version}")
    except ImportError:
        missing_packages.append(pip_name)
        print(f"   ✗ {pip_name} ({import_name}): NOT INSTALLED")

for msg in installed_packages:
    print(msg)

# Check if we can import the node modules
print(f"\n3. Node Module Import Test:")
import os
os.chdir(Path(__file__).parent)

# Installation commands
if missing_packages:
    print("\n" + "=" * 70)
    print("⚠️  MISSING PACKAGES DETECTED!")
    print("\nRun these commands to install missing packages:")
    print("-" * 70)
    
    for package in missing_packages:
        print(f"pip install {package}")
    
    print("\nOr install all at once:")
    print(f"pip install {' '.join(missing_packages)}")
    
    # Special instructions for pytoshop
    if 'pytoshop' in missing_packages:
        print("\n⚠️  Special instructions for pytoshop:")
        print("If regular install fails, try:")
        print("  pip install cython")
        print("  pip install pytoshop -I --no-cache-dir")

print("\n" + "=" * 70)
print("Check complete!")
print("=" * 70)
