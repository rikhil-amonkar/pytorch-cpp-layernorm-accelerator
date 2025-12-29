cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# project name
project(layernorm-systems)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)

# locate libtorch directory (adjust path to where you installed libtorch)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/Downloads/libtorch")

# Enable rpath on macOS
set(CMAKE_MACOSX_RPATH ON)

# find torch package in libtorch
find_package(Torch REQUIRED)

# set flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add executable files (path is relative to CMakeLists.txt location)
add_executable(multiply_two_tensors tests/tensor_memory_cpp/multiply_two_tensors.cpp)

# set target libraries
target_link_libraries(multiply_two_tensors "${TORCH_LIBRARIES}")

# Set rpath to find libtorch libraries at runtime
set_target_properties(multiply_two_tensors PROPERTIES
    CXX_STANDARD 17
    BUILD_RPATH "$ENV{HOME}/Downloads/libtorch/lib"
    INSTALL_RPATH "$ENV{HOME}/Downloads/libtorch/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)