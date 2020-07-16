#!/usr/bin/env python3
# coding: utf-8
"""Installation script for qsynthesis module."""

import sys
from setuptools import setup, find_packages

try:
    # Check that we have the appropriate Triton version
    import triton
    if triton.VERSION.BUILD < 1467:
        print("Triton >=0.8 is required")
        sys.exit(1)
except ImportError:
    print("Triton module should be installed first")
    sys.exit(1)

setup(
    name="qsynthesis",
    version="0.1",
    description="Python API to synthesize Triton AST's",
    packages=find_packages(),
    setup_requires=[],
    install_requires=["orderedset", "psutil", "click"],
    tests_require=[],
    license="qb",
    author="Quarkslab",
    classifiers=[
        'Topic :: Security',
        'Environment :: Console',
        'Operating System :: OS Independent',
    ],
    test_suite="",
    scripts=['bin/qsynthesis-table-manager']
)
