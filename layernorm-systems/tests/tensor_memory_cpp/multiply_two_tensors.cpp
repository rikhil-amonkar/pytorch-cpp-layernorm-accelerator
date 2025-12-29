#include <torch/torch.h>
#include <iostream>
using namespace std;  // pre-define namespace

torch::Tensor elementWiseMultiplication(torch::Tensor a, torch::Tensor b) {

    // validate input tensor contract (size, dtype, device, contiguous)
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensor objects are of unequal size.")  // element-wise multiplication can't occur with different tensor sizes
    TORCH_CHECK(a.dtype() == b.dtype(), "Input tensor objects are of unequal data type.")  // element-wise multiplication is messy with mixed data types
    TORCH_CHECK(a.device() == b.device(), "Input tensor objects are not stored on same device.")  // pointers should be on same device
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "One or more input tensor objects are not contiguous.")  // multiplying memory addresses (need same format)

    // define output tensor to store multiplied values
    torch::Tensor product = torch::empty_like(a);  // clone of tensor but different memory location

    // define data pointers for each tensor to refer to each element
    int *ptr_a = a.data_ptr<int>();
    int *ptr_b = b.data_ptr<int>();
    int *ptr_prod = product.data_ptr<int>();
    
    // declare total number of elements
    int n = a.numel();

    // iterate over all tensor elements
    for (int i = 0; i < n; i++) {
        
        // update element at output pointer location to product
        ptr_prod[i] = ptr_a[i] * ptr_b[i];

    }

    return product;

}

int main() {

    // create simple input tensors (a, b)
    torch::Tensor a = torch::rand({5, 3});
    torch::Tensor b = torch::rand({5, 3});

    // create complex (options) input tensors (c, d)
    torch::Tensor c = torch::randint(/*low*/1, /*high*/10, /*size*/{5, 3}, /*options*/torch::dtype(torch::kInt32).device(torch::kCPU));  // 32-bit signed int on CPU
    torch::Tensor d = torch::randint(/*low*/1, /*high*/10, /*size*/{5, 3}, /*options*/torch::dtype(torch::kInt32).device(torch::kCPU));  // 32-bit signed int on CPU

    // perform element-wise multiplication on tensors
    torch::Tensor product = elementWiseMultiplication(c, d);

    // display calculations
    cout << "\n================================" << endl;
    cout << "\nINPUT TENSOR A:\n" << c << endl;
    cout << "\nINPUT TENSOR B:\n" << d << endl;
    cout << "\nOUTPUT TENSOR (PRODUCT):\n" << product << endl;
    cout << "\n================================" << endl;

}