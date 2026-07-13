# Copyright (c) 2025-2026, Loco-Transformer Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Setup script for the loco_transformer package."""

from setuptools import setup, find_packages

setup(
    name="loco_transformer",
    version="1.0.0",
    packages=find_packages(include=["loco_transformer", "loco_transformer.*"]),
    python_requires=">=3.10",
)
