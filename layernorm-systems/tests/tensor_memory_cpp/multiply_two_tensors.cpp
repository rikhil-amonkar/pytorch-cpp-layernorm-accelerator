#include <torch/torch.h>
#include <iostream>
using namespace std;  // pre-define namespace

int main() {

    // create a simple tensor
    torch::Tensor tensor = torch::rand({5, 3});
    cout << tensor << endl;

}