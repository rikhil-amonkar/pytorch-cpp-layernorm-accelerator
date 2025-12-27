# High-Performance PyTorch LayerNorm via C++ Extension

## Overview

In this project, I will be building and documenting my journey into learning more about system-level ML and performance optimization. I will be creating a high-performance Layer Normalization operator as a custom PyTorch C++ extension, which will integrate forward and backward passes with PyTorch autograd, benchmark it against CPU and MPS backends, and possibly design it with a backend-agnostic architecture for future CUDA support if everything seems to go well. The goal is to gain hands-on experience with PyTorch internals, Python-C++ integration, and efficient neural network operator design.

## Dependencies (current)

Python 3.14.2
PyTorch 2.9.1

## Initial Setup Steps

**Python + PyTorch Setup**
- Make sure Python 3.10+ is installed
- PyTorch installed with MPS support (Apple GPU backend)
'''sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
'''

**XCode Command-Line Tools**
- Check if XCode is installed (look for 'Command line tools are already installed.' error)
- Needed for compiling C++ extensions on MacOS
'''sh
xcode-select --install
'''

**Ninja Package**
- Ninja is a small build system (like Make or CMake) used to compile C++ extensions quickly and efficiently
- Needed to run load() function with pytorch when calling the custom C++ modules
'''sh
pip install ninja
'''

## Dev Background

**Rikhil Amonkar | CS @ Drexel University**
**Experience:** ML Engineer Co-op @ Lockheed Martin | NLP Research Engineer @ Drexel University
**Contact me at:** rikhilma@gmail.com