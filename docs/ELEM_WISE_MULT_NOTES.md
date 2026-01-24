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

## Element-Wise Multiplication of Two Tensors in C++ (multiply_two_tensors.cpp)

### Contract (pre-defined rules)

Rules:
- Two tensors much have:
    - Same shape
    - Same data type (dtype)
    - Same device (stored)
    - Contiguous memory (same memory location)

### General Steps (Flow)

- Step 1: Receive two tensor handles
- Step 2: Verify contract assumptions
- Step 3: Allocate output tensor with correct metadata
- Step 4: Obtain raw memory pointers
- Step 5: Loop over total elements
- Step 6: Read from input A
- Step 7: Read from input B
- Step 8: Multiply
- Step 9: Write to output
- Step 10: Return output tensor

## Forward Pass Function in C++ (forward_pass_layernorm.cpp)

### Calculation Process Breakdowns

- Element-Wise Operations
    - Normalization: ```xhat = xmu * ivar```
    - Learnable Parameters: ```output = (gamma * xhat) + beta```
    - Explanation: Both of these operations are element-wise, meaning, they are tensors that can be flattened, which then allows for a data pointer to be used to access memory and calculation instead of multiple "for loops".
- Reductions (mean, sum, variance)
    - Mean: ```mu = (1 / dims) * np.sum(x, axis=-1, keepdims=True)```
    - Variance: ```var = (1 / dims) * np.sum(sq, axis=-1, keepdims=True)```
    - Explanation: Both of these operations need to be performed per row/sample group so they cannot be flattened. They require nested loops with one outer branch and one inner branch to go over the groups and then sub-features.



