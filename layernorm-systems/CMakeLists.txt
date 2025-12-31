cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# project name
project(layernorm-systems)

# set C++ standard
set(CMAKE_CXX_STANDARD 20)

# locate libtorch directory (adjust path to where you installed libtorch)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/Downloads/libtorch")

# Enable rpath on macOS
set(CMAKE_MACOSX_RPATH ON)

# find torch package in libtorch
find_package(Torch REQUIRED)

# set flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# import catch2 unit test framework
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)

# add executable files (path is relative to CMakeLists.txt location)
add_executable(multiply_two_tensors tests/tensor_memory_cpp/multiply_two_tensors.cpp)
add_library(forward_pass_layernorm_lib tests/tensor_memory_cpp/forward_pass_layernorm.cpp)
add_executable(unit_test tests/tensor_memory_cpp/unit_test.cpp)

# set target libraries
target_link_libraries(multiply_two_tensors "${TORCH_LIBRARIES}")
target_link_libraries(forward_pass_layernorm_lib "${TORCH_LIBRARIES}")
target_link_libraries(unit_test PRIVATE forward_pass_layernorm_lib "${TORCH_LIBRARIES}" Catch2::Catch2WithMain)

# Set rpath to find libtorch libraries at runtime
set_target_properties(multiply_two_tensors PROPERTIES
    CXX_STANDARD 20
    BUILD_RPATH "$ENV{HOME}/Downloads/libtorch/lib"
    INSTALL_RPATH "$ENV{HOME}/Downloads/libtorch/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_target_properties(forward_pass_layernorm_lib PROPERTIES
    CXX_STANDARD 20
    BUILD_RPATH "$ENV{HOME}/Downloads/libtorch/lib"
    INSTALL_RPATH "$ENV{HOME}/Downloads/libtorch/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_target_properties(unit_test PROPERTIES
    CXX_STANDARD 20
    BUILD_RPATH "$ENV{HOME}/Downloads/libtorch/lib"
    INSTALL_RPATH "$ENV{HOME}/Downloads/libtorch/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)


