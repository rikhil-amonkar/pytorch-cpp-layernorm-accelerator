# CMake Build System & Tensor Memory Test

## CMake Configuration (CMakeLists.txt)

### What CMake Does

- CMake is a build system generator that creates build files (Makefiles, etc.) to compile C++ code.
- CMakeLists.txt is the configuration file that tells CMake how to build your project.
- Unlike Python, C++ code must be compiled into an executable before you can run it.

### Key CMake Setup Steps

- Set minimum CMake version required (3.18).
- Define the project name (layernorm-systems).
- Set C++ standard version (C++17).
- Locate libtorch directory path where PyTorch C++ libraries are installed.
- Enable rpath on macOS so the executable can find libraries at runtime.
- Find the Torch package using find_package() which locates libtorch headers and libraries.
- Set compiler flags that PyTorch needs (includes TORCH_CXX_FLAGS automatically).
- Create an executable target named "multiply_two_tensors" from the source file.
- Link the executable with Torch libraries so it can use PyTorch functions.
- Set rpath properties so the executable knows where to find libtorch libraries when running.

### Build Workflow

1. Create build directory: `mkdir -p build && cd build`
2. Run CMake configuration: `cmake ..` (this only needs to be done once, or when CMakeLists.txt changes)
3. Compile the code: `make` (this needs to be run every time you change the .cpp file)
4. Run the executable: `./multiply_two_tensors`

### Important Notes

- This file must be recompiled (using `make`) every time you make changes to it.
- Unlike Python, C++ doesn't have a REPL - you must compile then run.
- The executable is created in the build directory after successful compilation.


