# Minimal PyTorch C++ Extension (Add Numbers)

## C++ Implementation (add_nums_pybind.cpp)
1. Create .cpp program with simple function and export as module
- Import the Pybind11 library which allows C++ code to be exposed as Python modules.
- Create a standard C++ function (such as taking two ints and returning the sum).
- Define the Python module with a name in C++ with 'm' being used as a module object to add functions and docstrings.
- Set a module-level docstring (like __doc__ in Python) to document the module and its purpose.
- Add the function to the Python module (name of function to call in Python, pointer to C++ function to expose, docstring).

## Python Implementation (add_nums_test.py)
2. Create a .py program to run the custom compiled module
- Import the PyTorch helper function to compile and load a C++ extension on the fly.
- Use the load function to compile the C++ file and return the Python module object (Python module name, C++ source files, compiler output).
- Call the C++ function from Python just like a normal Python function.


