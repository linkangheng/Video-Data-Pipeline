#!/usr/bin/python3
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#

import sys
import setuptools


if sys.version_info < (3, 6):
    sys.exit("Python versions less than 3.6 are not supported")

scripts = []

setuptools.setup(
    name="webdataset",
    version = "0.2.51",
    description="Record sequential storage for deep learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RunpeiDong/webdataset-private",
    author="Thomas Breuel, Runpei Dong",
    author_email="tmbdev+removeme@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="object store, client, deep learning",
    packages=["webdataset"],
    python_requires=">=3.6",
    scripts=scripts,
    install_requires="braceexpand numpy pyyaml".split(),
    license_files=["LICENSE"],
)
