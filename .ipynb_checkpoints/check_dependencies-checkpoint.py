#!/usr/bin/env python
"""
ComfyUI-fixableflow dependency check script
Run this to verify all requirements are properly installed
"""

import sys
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
}

missing_packages = []
installed_packages = []

for import_name, pip_name in packages.items():
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        installed_packages.append(f"   ? {pip_name} ({import_name}): {version}")
    except ImportError:
        missing_packages.append(pip_name)
        print(f"   ? {pip_name} ({import_name}): NOT INSTALLED")

for msg in installed_packages:
    print(msg)

# Check frontend bundle
print(f"\n3. Frontend Bundle:")
web_dir = Path(__file__).parent / "web"
ag_psd_bundle = web_dir / "ag-psd.bundle.js"
if ag_psd_bundle.exists():
    size_kb = ag_psd_bundle.stat().st_size / 1024
    print(f"   [OK] ag-psd.bundle.js exists ({size_kb:.1f} KB)")
else:
    print(f"   [X] ag-psd.bundle.js NOT FOUND")
    print(f"       Run: cd web-build && npm install && npm run build")

# Installation commands
if missing_packages:
    print("\n" + "=" * 70)
    print("MISSING PACKAGES DETECTED!")
    print("\nRun this command to install missing packages:")
    print("-" * 70)
    print(f"pip install {' '.join(missing_packages)}")

print("\n" + "=" * 70)
print("Note: pytoshop is no longer required!")
print("      PSD generation is now handled by frontend ag-psd library.")
print("=" * 70)
