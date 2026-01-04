#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "forward.h"

// define layernorm module and information
PYBIND11_MODULE(layernorm_module, m) {
    
    m.doc() = "layernorm c++ extension module";  // document full module

    // expose the forward output struct to python
    pybind11::class_<forwardOutput>(m, "ForwardOutput")
        .def_readonly("output", &forwardOutput::output)
        .def_readonly("cache", &forwardOutput::cache)
        .def_readonly("epsilon", &forwardOutput::epsilon);

    // bind function with keyword arguments
    m.def("forward", &forwardPassLayerNorm, 
        pybind11::arg("x"),
        pybind11::arg("gamma"),
        pybind11::arg("beta"),
        pybind11::arg("epsilon") = 1e-5f,  // default float epsilon
        "forward pass two tensors for layernorm");  // bind forward pass function

}
