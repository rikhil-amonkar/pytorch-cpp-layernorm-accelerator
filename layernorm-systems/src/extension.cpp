#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "forward.h"

// define layernorm module and information
PYBIND11_MODULE(layernorm_module, m) {
    
    m.doc() = "layernorm c++ extension module";  // document full module

    // expose the forward output struct to python
    pybind11::class_<forwardOutput>(m, "ForwardOutput")
        .def_readonly("output", &forwardOutput::output)
        .def_readonly("mu", &forwardOutput::mu)
        .def_readonly("var", &forwardOutput::var)
        .def_readonly("std", &forwardOutput::std)
        .def_readonly("ivar", &forwardOutput::ivar)
        .def_readonly("cache_tensors", &forwardOutput::cache_tensors)
        .def_readonly("epsilon", &forwardOutput::epsilon);

    // bind function with keyword arguments
    m.def("forward", &forwardPassLayerNorm, 
        pybind11::arg("x"),
        pybind11::arg("gamma"),
        pybind11::arg("beta"),
        pybind11::arg("epsilon") = 1e-5f,  // default float epsilon
        "forward pass two tensors for layernorm");  // bind forward pass function

}
