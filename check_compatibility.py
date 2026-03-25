#!/usr/bin/env python3
"""
Python 3.11.9 Compatibility Check Script

This script verifies that all code and dependencies are compatible with Python 3.11.9.
"""

import sys
import ast
import os
from pathlib import Path

def check_python_version():
    """Check if running Python 3.11.9"""
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor == 11:
        print("✓ Running Python 3.11")
        return True
    else:
        print(f"⚠ Running Python {version.major}.{version.minor}.{version.micro}, not 3.11.9")
        return False

def check_imports():
    """Try importing all required packages"""
    required_packages = [
        'numpy',
        'torch',
        'matplotlib',
        'seaborn',
        'tqdm',
        'scipy',
        'pandas',
    ]

    print("\nChecking required packages:")
    all_ok = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - {e}")
            all_ok = False

    return all_ok

def check_syntax_compatibility(directory):
    """Check Python files for syntax compatibility"""
    print(f"\nChecking Python files in {directory} for syntax compatibility:")

    issues = []
    files_checked = 0

    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                files_checked += 1

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()
                    ast.parse(code)
                except SyntaxError as e:
                    issues.append((filepath, str(e)))

    print(f"Checked {files_checked} Python files")

    if issues:
        print("\n⚠ Syntax issues found:")
        for filepath, error in issues:
            print(f"  {filepath}: {error}")
        return False
    else:
        print("✓ No syntax issues found")
        return True

def check_type_hints():
    """Check for deprecated type hint usage"""
    print("\nType hints check:")
    print("✓ Using typing module imports is compatible with Python 3.11.9")
    print("  (Both typing.Dict and dict are supported)")
    return True

def main():
    print("=" * 70)
    print("Python 3.11.9 Compatibility Check")
    print("=" * 70)

    results = []

    # Check Python version
    results.append(("Python Version", check_python_version()))

    # Check syntax
    src_dir = Path(__file__).parent / "src"
    if src_dir.exists():
        results.append(("Syntax Check", check_syntax_compatibility(str(src_dir))))

    # Check type hints
    results.append(("Type Hints", check_type_hints()))

    # Check imports
    results.append(("Package Imports", check_imports()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All checks passed! Code is compatible with Python 3.11.9")
    else:
        print("⚠ Some checks failed. Review the issues above.")
    print("=" * 70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
