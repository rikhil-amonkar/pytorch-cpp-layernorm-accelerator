#include <torch/torch.h>
#include <iostream>
using namespace std;  // pre-define namespace

// create template to run multiplication with any data type
template <typename T>
void multiplyElements(torch::Tensor a, torch::Tensor b, torch::Tensor product) {

    // define data pointers for each tensor to refer to each element
    T *ptr_a = a.data_ptr<T>();
    T *ptr_b = b.data_ptr<T>();
    T *ptr_prod = product.data_ptr<T>();

    // declare total number of elements
    int n = a.numel();

    // iterate over all tensor elements
    for (int i = 0; i < n; i++) {
        
        // update element at output pointer location to product
        ptr_prod[i] = ptr_a[i] * ptr_b[i];

    }

}

torch::Tensor elementWiseMultiplication(torch::Tensor a, torch::Tensor b) {

    // validate input tensor contract (size, dtype, device, contiguous)
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensor objects are of unequal size.")  // element-wise multiplication can't occur with different tensor sizes
    TORCH_CHECK(a.dtype() == b.dtype(), "Input tensor objects are of unequal data type.")  // element-wise multiplication is messy with mixed data types
    TORCH_CHECK(a.device() == b.device(), "Input tensor objects are not stored on same device.")  // pointers should be on same device
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "One or more input tensor objects are not contiguous.")  // multiplying memory addresses (need same format)

    // define output tensor to store multiplied values
    torch::Tensor product = torch::empty_like(a);  // clone of tensor but different memory location

    // select data type to multiply based on tensor scalar type
    switch (a.scalar_type()) {

        // case for int type
        case torch::kInt32 : {
            multiplyElements<int32_t>(a, b, product);
            break;
        }

        // case for float type
        case torch::kFloat32 : {
            multiplyElements<float>(a, b, product);
            break;
        }

        // default case for unsupported data types
        default : {
            TORCH_CHECK(false, "Unsupported dtype: ", a.dtype(), ". Only kInt32 and kFloat32 are supported.")
        }

    }

    return product;  // return updated output tensor object

}

int main() {

    // create complex (options) input tensors (a, b)
    torch::Tensor a = torch::randint(/*low*/1, /*high*/10, /*size*/{5, 3}, /*options*/torch::dtype(torch::kInt32).device(torch::kCPU));  // 32-bit signed int on CPU
    torch::Tensor b = torch::randint(/*low*/1, /*high*/10, /*size*/{5, 3}, /*options*/torch::dtype(torch::kInt32).device(torch::kCPU));  // 32-bit signed int on CPU

    // perform element-wise multiplication on tensors
    torch::Tensor product = elementWiseMultiplication(a, b);

    // display calculations
    cout << "\n================================" << endl;
    cout << "\nINPUT TENSOR A:\n" << a << endl;
    cout << "\nINPUT TENSOR B:\n" << b << endl;
    cout << "\nOUTPUT TENSOR (PRODUCT):\n" << product << endl;
    cout << "\n================================" << endl;

}