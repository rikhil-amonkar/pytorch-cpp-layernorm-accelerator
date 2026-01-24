#include <iostream>
#include <torch/torch.h>
using namespace std;

auto checkDevice(torch::Tensor x) {

    auto device = x.device();

    return device;

}

int main() {

    torch::Tensor x = torch::randn({3, 3});
    auto device = checkDevice(x);
    
    if (device.is_cpu()) {
        cout << "CPU Tensor" << endl;
    } else if (device.is_cuda()) {
        cout << "CUDA Tensor located on device " << device.index() << endl;
    }

    return 0;

}