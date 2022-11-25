#!/usr/bin/env python3
# coding: utf-8
"""Installation script for qsynthesis module."""

from setuptools import setup, find_packages

generate_deps = ['sympy']
server_deps = ['fastapi', 'uvicorn']
assembly_deps = ["arybo", "llvmlite"]

setup(
    name="qsynthesis",
    version="0.1.1",
    description="Python API to synthesize Triton AST's",
    long_description_content_type='text/markdown',
    long_description="file: README.md",
    packages=find_packages(),
    url="https://github.com/quarkslab/qsynthesis",
    project_urls={
        "Documentation": "https://quarkslab.github.io/qsynthesis/",
        "Bug Tracker": "https://github.com/quarkslab/qsynthesis/issues",
        "Source": "https://github.com/quarkslab/qsynthesis"
    },
    setup_requires=[],
    install_requires=["triton-library",
                      "ordered_set",
                      "psutil",
                      "click",
                      "plyvel",
                      "requests",
                      "capstone",
                      "pydffi>=0.9.1"],
    tests_require=[],
    license="AGPL-3.0",
    author="Robin David",
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
