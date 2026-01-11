#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "backward.h"

using namespace std;

// function definition for performing backward pass on output and cache via layernorm
backwardOutput backwardPassLayerNorm(torch::Tensor dout, vector<torch::Tensor> cache, double epsilon) {

    // extract intermediate values from forward pass cache (also guarantee row-major)
    torch::Tensor gamma = cache[0].contiguous();
    torch::Tensor xhat = cache[1].contiguous();
    torch::Tensor xmu = cache[2].contiguous();
    torch::Tensor sqrtvar = cache[3].contiguous();
    torch::Tensor ivar = cache[4].contiguous();
    torch::Tensor var = cache[5].contiguous();

    // validate forward pass output with backward pass input tensor (contract)
    TORCH_CHECK(dout.sizes() == xhat.sizes(), "Input tensor does not match the correct element size.");  // element-wise ops
    TORCH_CHECK(dout.dtype() == xhat.dtype(), "Input tensor has an invalid data types.");  // data type (for pointer)
    TORCH_CHECK(dout.device() == xhat.device(), "Input tensor is not located on the correct device.");  // memory location
    TORCH_CHECK(dout.is_contiguous(), "Input tensor is not contiguous.");  // data layout order

    // get dimensions of tensor (input/output)
    int dims = dout.size(-1);  // last dim
    int n = dout.numel() / dims;  // elements

    // create backpass output tensors
    torch::Tensor dx = torch::zeros_like(dout);
    torch::Tensor dgamma = torch::zeros({dims}, dout.options());  // accumulator (gamma)
    torch::Tensor dbeta = torch::zeros({dims}, dout.options());  // accumulator (beta)

    // initiate data pointers for output tensors (element-wise ops)
    double *ptr_dx = dx.data_ptr<double>();  // 2-D (N, D)
    double *ptr_dgamma = dgamma.data_ptr<double>();  // 1-D (D,)
    double *ptr_dbeta = dbeta.data_ptr<double>();  // 1-D (D,)
    
    // initiate data pointers for cache tensors (element-wise ops)
    double *ptr_dout = dout.data_ptr<double>();  // 2-D (N, D)
    double *ptr_gamma = gamma.data_ptr<double>();  // 1-D (D,)
    double *ptr_xhat = xhat.data_ptr<double>();  // 2-D (N, D)
    double *ptr_xmu = xmu.data_ptr<double>();  // 2-D (N, D)
    double *ptr_sqrtvar = sqrtvar.data_ptr<double>();  // 1-D (N,)
    double *ptr_ivar = ivar.data_ptr<double>();  // 1-D (N,)
    double *ptr_var = var.data_ptr<double>();  // 1-D (N,)

    // create intermediate ops tensors (temp)
    torch::Tensor dxhat = torch::empty_like(xhat);
    torch::Tensor divar = torch::empty({n}, xhat.options());  // 1-D but xhat parameters
    torch::Tensor dxmu1 = torch::empty_like(xhat);
    torch::Tensor dsqrtvar = torch::empty({n}, xhat.options());  // 1-D but xhat parameters
    torch::Tensor dvar = torch::empty({n}, xhat.options());  // 1-D but xhat parameters
    torch::Tensor dsq = torch::empty_like(xhat);
    torch::Tensor dxmu2 = torch::empty_like(xhat);
    torch::Tensor dx1 = torch::empty_like(xhat);
    torch::Tensor dmu = torch::empty({n}, xhat.options());  // 1-D but xhat parameters
    torch::Tensor dx2 = torch::empty_like(xhat);

    // initiate data pointers for intermediate ops tensors
    double *ptr_dxhat = dxhat.data_ptr<double>();  // 2-D (N, D)
    double *ptr_divar = divar.data_ptr<double>();  // 1-D (N,)
    double *ptr_dxmu1 = dxmu1.data_ptr<double>();  // 2-D (N, D)
    double *ptr_dsqrtvar = dsqrtvar.data_ptr<double>();  // 1-D (N,)
    double *ptr_dvar = dvar.data_ptr<double>();  // 1-D (N,)
    double *ptr_dsq = dsq.data_ptr<double>();  // 2-D (N, D)
    double *ptr_dxmu2 = dxmu2.data_ptr<double>();  // 2-D (N, D)
    double *ptr_dx1 = dx1.data_ptr<double>();  // 2-D (N, D)
    double *ptr_dmu = dmu.data_ptr<double>();  // 1-D (N,)
    double *ptr_dx2 = dx2.data_ptr<double>();  // 2-D (N, D)

    // iterate through rows/groups
    for (int i = 0; i < n; i++) {

        // linear scaling for learnable parameters (gradients)
        for (int j = 0; j < dims; j++) {
            ptr_dbeta[j] += ptr_dout[(i * dims) + j];  // dbeta
            ptr_dgamma[j] += (ptr_dout[(i * dims) + j] * ptr_xhat[(i * dims) + j]);  // dgamma
            ptr_dxhat[(i * dims) + j] = (ptr_dout[(i * dims) + j] * ptr_gamma[j]);  // dxhat
        }

        // calculate gradient variance (w.r.t.) inverse variance
        double divar_sum = 0.0f;
        for (int j = 0; j < dims; j++) {
            divar_sum += (ptr_dxhat[(i * dims) + j] * ptr_xmu[(i * dims) + j]);  // var accumulator
        }
        ptr_divar[i] = divar_sum;  // divar

        // calculate first component of gradient (w.r.t.) centered mean
        for (int j = 0; j < dims; j++) {
            ptr_dxmu1[(i * dims) + j] = (ptr_dxhat[(i * dims) + j] * ptr_ivar[i]);  // dxmu1
        }

        // calculate gradient (w.r.t.) squared variance via chain rule
        ptr_dsqrtvar[i] = (-1.0f / (ptr_sqrtvar[i] * ptr_sqrtvar[i])) * ptr_divar[i];  // dsqrtvar

        // calculate gradient (w.r.t.) variance via chain rule
        ptr_dvar[i] = 0.5f * (1.0f / ptr_sqrtvar[i]) * ptr_dsqrtvar[i];  // dvar

        // calculate gradient (w.r.t.) squared deviations via chain rule
        for (int j = 0; j < dims; j++) {
            ptr_dsq[(i * dims) + j] = (1.0f / dims) * ptr_dvar[i];  // dsq
        }

        // calculate components of gradient (w.r.t.) centered mean
        for (int j = 0; j < dims; j++) {
            ptr_dxmu2[(i * dims) + j] = (2.0f * ptr_xmu[(i * dims) + j] * ptr_dsq[(i * dims) + j]);  // dxmu2
            ptr_dx1[(i * dims) + j] = (ptr_dxmu1[(i * dims) + j] + ptr_dxmu2[(i * dims) + j]);  // dx1
        }

        // calculate gradient (w.r.t.) mean
        double dmu_sum = 0.0f;
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
