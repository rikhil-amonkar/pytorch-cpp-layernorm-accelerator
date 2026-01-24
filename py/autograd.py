import torch
from torch.utils.cpp_extension import load

optimize = True

if optimize:

    # path to c++ extensions
    CPP_PATH = "./optimized"

    # load layernorm c++ extensions
    lnext = load(
        name="layernorm_module",  # module
        sources=[
            f"{CPP_PATH}/extension.cpp",  # signature
            f"{CPP_PATH}/forward.cpp",  # forward def
            f"{CPP_PATH}/backward.cpp"  # backward def
        ],
        verbose=True  # compiler
    )

    # autograd function class (inherits from pytorch autograd)
    class LayerNorm(torch.autograd.Function):
        
        # define static forward method (no-instance needed)
        @staticmethod
        def forward(ctx, x, gamma, beta, epsilon=1e-5):
            
            # call forward pass extension
            result = lnext.forward(x, gamma, beta, epsilon)
            output, cache = result.output, result.cache  # unpack
            gamma, xhat, xmu, sqrtvar, ivar = cache  # seperate cache
            
            # save intermediates in context for backward pass
            ctx.save_for_backward(gamma, xhat, xmu, sqrtvar, ivar)
            ctx.epsilon = 1e-5  # store epsilon
            
            return output
        
        # define static backward method (no-instance needed)
        @staticmethod
        def backward(ctx, dout):
                    
            # retrieve cache from context
            gamma, xhat, xmu, sqrtvar, ivar, = ctx.saved_tensors  # unpack
            cache = gamma, xhat, xmu, sqrtvar, ivar  # store
            epsilon = ctx.epsilon  # extract epsilon
            dout = dout.contiguous()  # convert to row-major
            
            # call backward pass extension
            result = lnext.backward(dout, cache, epsilon)
            dx, dgamma, dbeta = result.dx, result.dgamma, result.dbeta  # unpack
            
            return dx, dgamma, dbeta, None
        
else:

    # path to c++ extensions
    CPP_PATH = "./cpp"

    # load layernorm c++ extensions
    lnext = load(
        name="layernorm_module",  # module
        sources=[
            f"{CPP_PATH}/extension.cpp",  # signature
            f"{CPP_PATH}/forward.cpp",  # forward def
            f"{CPP_PATH}/backward.cpp"  # backward def
        ],
        verbose=True  # compiler
    )

    # autograd function class (inherits from pytorch autograd)
    class LayerNorm(torch.autograd.Function):
        
        # define static forward method (no-instance needed)
        @staticmethod
        def forward(ctx, x, gamma, beta, epsilon=1e-5):
            
            # call forward pass extension
            result = lnext.forward(x, gamma, beta, epsilon)
            output, cache = result.output, result.cache  # unpack
            gamma, xhat, xmu, sqrtvar, ivar, var = cache  # seperate cache
            
            # save intermediates in context for backward pass
            ctx.save_for_backward(gamma, xhat, xmu, sqrtvar, ivar, var)
            ctx.epsilon = 1e-5  # store epsilon
            
            return output
        
        # define static backward method (no-instance needed)
        @staticmethod
        def backward(ctx, dout):
                    
            # retrieve cache from context
            gamma, xhat, xmu, sqrtvar, ivar, var, = ctx.saved_tensors  # unpack
            cache = gamma, xhat, xmu, sqrtvar, ivar, var  # store
            epsilon = ctx.epsilon  # extract epsilon
            dout = dout.contiguous()  # convert to row-major
            
            # call backward pass extension
            result = lnext.backward(dout, cache, epsilon)
            dx, dgamma, dbeta = result.dx, result.dgamma, result.dbeta  # unpack
            
            return dx, dgamma, dbeta, None