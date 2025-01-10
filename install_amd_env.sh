#!/bin/bash -i
micromamba env create -f environment-amd.yml -y
micromamba run -n igdesign pip install "torch>=2.5,<2.6" --index-url https://download.pytorch.org/whl/rocm6.2 --force-reinstall
micromamba run -n igdesign pip install -e .