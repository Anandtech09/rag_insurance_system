#!/usr/bin/env python3
"""
Setup configuration for Insurance RAG System
"""

from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="insurance-rag-system",
    version="1.0.0",
    description="Multi-Document RAG System for Insurance Policy Analysis",
    author="Insurance RAG Team",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'insurance-rag=scripts.run_system:main',
            'insurance-demo=scripts.demo:main',
            'insurance-test=scripts.test_system:main',
        ],
    },
)