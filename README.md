# High-Performance PyTorch LayerNorm via C++ Extension

## Dependencies

Python + PyTorch Setup
- Make sure Python 3.10+ is installed
- PyTorch installed with MPS support (Apple GPU backend)
'''sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
'''

XCode Command-Line Tools
- Needed for compiling C++ extensions on MacOS
'''sh
xcode-select --install
'''

Ninja Package
- Ninja is a small build system (like Make or CMake) used to compile C++ extensions quickly and efficiently (needed to run load() function with torch)
'''sh
pip install ninja
'''