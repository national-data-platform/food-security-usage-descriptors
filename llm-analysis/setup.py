#!/usr/bin/env python3
"""
Backward compatibility setup.py for pub-analysis-agent.

This file exists for compatibility with tools that require setup.py.
All configuration is defined in pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup() 