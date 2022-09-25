#!/usr/bin/env python3
# coding: utf-8
"""Installation script for qsynthesis module."""

from setuptools import setup, find_packages

generate_deps = ['pydffi>=0.9.1', 'sympy']
server_deps = ['fastapi', 'uvicorn']
assembly_deps = ["arybo", "llvmlite"]

setup(
    name="qsynthesis",
    version="0.1",
    description="Python API to synthesize Triton AST's",
    packages=find_packages(),
    setup_requires=[],
    install_requires=["triton-library", "ordered_set", "psutil", "click", "plyvel", "requests", "capstone"],
    tests_require=[],
    license="qb",
    author="Quarkslab",
    classifiers=[
        'Topic :: Security',
        'Environment :: Console',
        'Operating System :: OS Independent',
    ],
    extras_require={
        'all': assembly_deps+generate_deps+server_deps,
        'reassembly': assembly_deps,
        'generator': generate_deps,
        'server': server_deps
    },
    test_suite="",
    scripts=['bin/qsynthesis-table-manager', 'bin/qsynthesis-table-server']
)
