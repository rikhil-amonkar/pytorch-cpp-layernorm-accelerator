from torch.utils.cpp_extension import load

# define path to cpp file
CPP_PATH = "./layernorm-systems/tests/compile_minimal_extension/add_nums_pybind.cpp"

# load in cpp extension module
cpp_ext = load(
    
    name="pybind_example_module",  # module name
    sources=[CPP_PATH],  # cpp filename
    verbose=True  # output to compiler for debug
    
)

# run a short test with two test ints and the compiled module
a, b = 2, 3  # test ints
sum_res = cpp_ext.add(a, b)  # call addition function from module (test)
avg_res = cpp_ext.average(a, b)  # call average function (test)
print(f"Addition of {a} + {b} results in a sum of: {sum_res}")
print(f"Averaging of {a} and {b} results in an average of: {avg_res}")