#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "backward_pass_layernorm.h"
#include "../forward_pass/forward_pass_layernorm.h"
using namespace std;

// forward pass output (for ref) = {output, (gamma, xhat, xmu, sqrtvar, ivar, var), epsilon}

// function definition for performing backward pass on output and cache via layernorm
backwardOutput backwardPassLayerNorm(torch::Tensor dout, vector<torch::Tensor> cache, float epsilon) {

    // extract intermediate values from forward pass cache
    torch::Tensor gamma = cache[0];
    torch::Tensor xhat = cache[1];
    torch::Tensor xmu = cache[2];
    torch::Tensor sqrtvar = cache[3];
    torch::Tensor ivar = cache[4];
    torch::Tensor var = cache[5];

    // validate forward pass output with backward pass input tensor (contract)
    TORCH_CHECK(dout.sizes() == xhat.sizes(), "Input tensor does not match the correct element size.");  // element-wise ops
    TORCH_CHECK(dout.dtype() == xhat.dtype(), "Input tensor has an invalid data types.");  // data type (for pointer)
    TORCH_CHECK(dout.device() == xhat.device(), "Input tensor is not located on the correct device.");  // memory location
    TORCH_CHECK(dout.is_contiguous(), "Input tensor is not contiguous.");  // data layout order

    // get dimensions of tensor (input/output)
    int dims = dout.size(-1);  // last dim
    int n = dout.numel() / dims;  // elements

    // create backpass output tensors
    torch::Tensor dx = torch::empty_like(dout);
    torch::Tensor dgamma = torch::zeros(dims);  // accumulator (gamma)
    torch::Tensor dbeta = torch::zeros(dims);  // accumulator (beta)

    // initiate data pointers for output tensors (element-wise ops)
    float *ptr_dx = dx.data_ptr<float>();  // 2-D (N, D)
    float *ptr_dgamma = dgamma.data_ptr<float>();  // 1-D (D,)
    float *ptr_dbeta = dbeta.data_ptr<float>();  // 1-D (D,)
    
    // initiate data pointers for cache tensors (element-wise ops)
    float *ptr_dout = dout.data_ptr<float>();  // 2-D (N, D)
    float *ptr_gamma = gamma.data_ptr<float>();  // 1-D (D,)
    float *ptr_xhat = xhat.data_ptr<float>();  // 2-D (N, D)
    float *ptr_xmu = xmu.data_ptr<float>();  // 2-D (N, D)
    float *ptr_sqrtvar = sqrtvar.data_ptr<float>();  // 1-D (N,)
    float *ptr_ivar = ivar.data_ptr<float>();  // 1-D (N,)
    float *ptr_var = var.data_ptr<float>();  // 1-D (N,)

    // create intermediate ops tensors (temp)
    torch::Tensor dxhat = torch::empty_like(xhat);
    torch::Tensor divar = torch::empty_like(ivar);
    torch::Tensor dxmu1 = torch::empty_like(xhat);
    torch::Tensor dsqrtvar = torch::empty_like(sqrtvar);
    torch::Tensor dvar = torch::empty_like(var);
    torch::Tensor dsq = torch::empty_like(var);
    torch::Tensor dxmu2 = torch::empty_like(xhat);
    torch::Tensor dx1 = torch::empty_like(dxmu2);
    torch::Tensor dmu = torch::empty_like(dx1);
    torch::Tensor dx2 = torch::empty_like(dmu);

    // initiate data pointers for intermediate ops tensors
    float *ptr_dxhat = dxhat.data_ptr<float>();  // 2-D (N, D)
    float *ptr_divar = divar.data_ptr<float>();  // 1-D (N,)
    float *ptr_dxmu1 = dxmu1.data_ptr<float>();  // 2-D (N, D)
    float *ptr_dsqrtvar = dsqrtvar.data_ptr<float>();  // 1-D (N,)
    float *ptr_dvar = dvar.data_ptr<float>();  // 1-D (N,)
    float *ptr_dsq = dsq.data_ptr<float>();  // 2-D (N, D)
    float *ptr_dxmu2 = dxmu2.data_ptr<float>();  // 2-D (N, D)
    float *ptr_dx1 = dx1.data_ptr<float>();  // 2-D (N, D)
    float *ptr_dmu = dmu.data_ptr<float>();  // 1-D (N,)
    float *ptr_dx2 = dx2.data_ptr<float>();  // 2-D (N, D)

    // iterate through rows/groups
    for (int i = 0; i < n; i++) {

        // linear scaling for learnable parameters (gradients)
        for (int j = 0; j < dims; j++) {
            ptr_dxhat[(i * dims) + j] = (ptr_dout[(i * dims) + j] * ptr_gamma[j]);  // dxhat
            ptr_dbeta[j] += ptr_dout[(i * dims) + j];  // dbeta
            ptr_dgamma[j] += (ptr_dout[(i * dims) + j] * ptr_xhat[(i * dims) + j]);  // dgamma
        }

        // calculate gradient variance (w.r.t.) inverse variance
        float divar_sum = 0.0f;
        for (int j = 0; j < dims; j++) {
            divar_sum += (ptr_dxhat[(i * dims) + j] * ptr_xmu[(i * dims) + j]);  // var accumulator
        }
        ptr_divar[i] = divar_sum;  // divar

        // calculate first component of gradient (w.r.t.) centered mean
        for (int j = 0; j < dims; j++) {
            ptr_dxmu1[(i * dims) + j] = (ptr_dxhat[(i * dims) + j] * ptr_ivar[i]);  // dxmu1
        }

        // calculate gradient (w.r.t.) squared variance via chain rule
        ptr_dsqrtvar[i] = (-1 / (ptr_sqrtvar[i] * ptr_sqrtvar[i])) * ptr_divar[i];  // dsqrtvar

        // calculate gradient (w.r.t.) variance via chain rule
        ptr_dvar[i] = 0.5 * (1 / sqrt((ptr_var[i] + epsilon))) * ptr_dsqrtvar[i];  // dvar

        // calculate gradient (w.r.t.) squared deviations via chain rule
        for (int j = 0; j < dims; j++) {
            ptr_dsq[(i * dims) + j] = (1.0f / dims) * ptr_dvar[i];  // dsq
        }

        // calculate components of gradient (w.r.t.) centered mean
        for (int j = 0; j < dims; j++) {
            ptr_dxmu2[(i * dims) + j] = (2 * ptr_xmu[(i * dims) + j] * ptr_dsq[(i * dims) + j]);  // dxmu2
            ptr_dx1[(i * dims) + j] = (ptr_dxmu1[(i * dims) + j] + ptr_dxmu2[(i * dims) + j]);  // dx1
        }

        // calculate gradient (w.r.t.) mean
        float dmu_sum = 0.0f;
        for (int j = 0; j < dims; j++) {
            dmu_sum += ptr_dx1[(i * dims) + j];  // mean accumulator
        }
        ptr_dmu[i] = -1.0f * dmu_sum;  // dmu

        // calculate components of gradient (w.r.t.) mean computations
        for (int j = 0; j < dims; j++) {
            ptr_dx2[(i * dims) + j] = (1.0f / dims) * ptr_dmu[i];  // dx2
        }

        // calculate final gradient using components (w.r.t.) input
        for (int j = 0; j < dims; j++) {
            ptr_dx[(i * dims) + j] = (ptr_dx1[(i * dims) + j] + ptr_dx2[(i * dims) + j]);  // dx
    
        }

    }

    return {dx, dgamma, dbeta};  // return outputs (3 tensors)

}

int main() {
    
    // create sample input tensor
    int n_samples = 5, embedding_dims = 3;
    torch::Tensor x = torch::rand({n_samples, embedding_dims}, torch::dtype(torch::kFloat32).device(torch::kCPU));  // 5x3, float32, cpu
    torch::Tensor gamma = torch::ones(embedding_dims);  // gamma
    torch::Tensor beta = torch::zeros(embedding_dims);  // beta 

    cout << "\nForward Pass Function (LayerNorm C++)!" << endl;

    // call forward pass
    forwardOutput result_forward = forwardPassLayerNorm(x, gamma, beta);  // struct result type

    // extract mean and std to check values across first row
    float mean_f = result_forward.output[0].mean().item<float>();
    float std_f = result_forward.output[0].std().item<float>();

    // print results
    cout << "\n===================================" << endl;
    cout << "Input Tensor SIZE: " << x.sizes() << endl;
    cout << format("FW Output Tensor MEAN: {:.4f} (should be ~0)", mean_f) << endl;
    cout << format("FW Output Tensor STD: {:.4f} (should be ~1)", std_f) << endl;
    cout << "===================================\n" << endl;

    cout << "\nBackward Pass Function (LayerNorm C++)!" << endl;

    // forward pass output (for ref) = {output, gamma, xhat, xmu, sqrtvar, ivar, var, epsilon}

    // create sample backpass dout input tensor
    torch::Tensor dout = torch::randn_like(result_forward.output);

    // call backward pass
    backwardOutput result_backward = backwardPassLayerNorm(dout, result_forward.cache, result_forward.epsilon);

    // display the results of the backward pass magnitudes
    cout << "\n===================================" << endl;
    cout << "Input Tensor SIZE: " << x.sizes() << endl;
    cout << "BW Output Tensor (dx):\n" << result_backward.dx << endl;
    cout << "BW Output Tensor (dgamma):\n" << result_backward.dgamma << endl;
    cout << "BW Output Tensor (dbeta):\n" << result_backward.dbeta << endl;
    cout << "===================================\n" << endl;

    return 0;

}
