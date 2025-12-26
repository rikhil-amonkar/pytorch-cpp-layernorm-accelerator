#include <iostream>
#include <pybind11/pybind11.h>

// add two integer values and return sum
int add(int i, int j) {

    std::cout << "Adding some numbers..." << "\n";  // Test statement
    return i + j;  // sum ints and return

}

// average two integer values and return result
double average(double i, double j) {

    std::cout << "Hello, this is a test for averaging." << "\n";  // Test statement
    return (i + j) / 2;

}

// define module name and parameters
PYBIND11_MODULE(pybind_example_module, m) {

    m.doc() = "pybind11 example";  // document module
    m.def("add", &add, "add two int numbers");  // py func name, cpp func name, docstring (addition function)
    m.def("average", &average, "average two int numbers");  // another function (average function added to module)
    
}