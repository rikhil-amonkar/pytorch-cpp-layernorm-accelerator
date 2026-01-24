# Architecture of Optimized LayerNorm Programs

## Overview

This folder contains C++ programs which all contribute to the projects main goal of creating a custom PyTorch extension in C++ with future integration into CUDA, while also testing the computing power of CPU vs MPS. These programs have been drafted in other directories, but have been seperated as they are the most cleaned and optimized versions, targeting good code practices such as excessive comments (mainly to learn), and organization between functions and variable names. The programs have also been continuously unit tested against the built-in PyTorch from both C++ using the LibTorch library, and also inside of Python itself, calling the custom extension as an autograd function. The two CUDA files that have been made serve as templates for future CUDA use, as this project was used as a way to learn about the whole system of Layernorm, the backbone of the C++ language and its connection to memory usage, and lastly, the basics of CUDA and NVIDIA GPU's, and how they can be used to compute functions in parallel. Sadly, access to an NVIDIA Driver and the GPU was limited for this project, so it was cut off short from that part, but luckily a lot was still done and learned. Future use will include implementation of CUDA to evaluate new speeds and benchmarks of the custom C++ PyTorch Layernorm extension.

## Files

- Backward Pass Functions (loss, cache, gradients)
    - backward.cpp
    - backward.h
- Forward Pass Functions (input tensors, learnable parameters, epsilon)
    - forward.cpp
    - forward.h
- CUDA Templates for Forward and Backward Pass Functions (for future)
    - backward.cu
    - forward.cu