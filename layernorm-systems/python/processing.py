import torch
from torch.utils.benchmark import Timer
        
if __name__ == "__main__":
    
    # sample test cases (baseline)
    cases = [
        (100, 512),
        (1000, 1024),
        (5000, 256),
        (10000, 10000)  # super big quick test
    ]
    
    # iterate through cases
    for case in cases:
        
        # create sample data (track gradients)
        n, dims = case
        device = torch.device("cpu")
        # device = torch.device("mps")

        # pt input tensor data
        pt_x = torch.rand((n, dims), dtype=torch.float32, requires_grad=True)  # input
        pt_target = torch.rand((n, dims), dtype=torch.float32)  # target for loss
        
        # manual input tensor data
        m_x = pt_x.clone().detach().to(torch.float32).requires_grad_(True)  # input (leaf)
        m_gamma = torch.ones((dims), dtype=torch.float32, requires_grad=True)  # scale
        m_beta = torch.zeros((dims), dtype=torch.float32, requires_grad=True)  # shift 
        m_target = pt_target.clone().detach().to(torch.float32)  # target for loss (leaf)
        
        # move to device memory
        # pt_x = pt_x.to(device)
        # pt_target = pt_target.to(device)
        m_x = m_x.to(device)
        m_gamma = m_gamma.to(device)
        m_beta = m_beta.to(device)
        m_target = m_target.to(device)       
        
        # # create timer instance
        # timer = Timer(
        #     stmt="""
        #     import torch
        #     import torch.nn as nn
            
        #     # limit randomness
        #     torch.manual_seed(42)

        #     def torchLayerNorm(dims, pt_x, pt_target):
                
        #         # pytorch layer normalization function
        #         ptln = nn.LayerNorm(dims).to(pt_x.device)

        #         # reset pytorch learn-param grads
        #         with torch.no_grad():
        #             ptln.weight.data = torch.ones(dims, dtype=torch.float32, device=pt_x.device, requires_grad=True)  # pt gamma
        #             ptln.bias.data = torch.zeros(dims, dtype=torch.float32, device=pt_x.device, requires_grad=True)  # pt beta
                    
        #         # pass tensor through pt forward layernorm
        #         ptout = ptln(pt_x)

        #         # calculate pt loss (mse)
        #         ptloss = ((ptout - pt_target) ** 2).sum()  # mean sq err sum

        #         # pass loss grad through pt backward layernorm
        #         ptloss.backward()
            
        #     torchLayerNorm(dims, pt_x, pt_target)
        #     """,
        #     globals={"dims": dims, "pt_x": pt_x, "pt_target": pt_target}
        # )
        
        timer = Timer(
            stmt="""
            import torch
            import torch.nn as nn
            from autograd import LayerNorm
            
            # limit randomness
            torch.manual_seed(42)

            def manualLayerNorm(m_x, m_gamma, m_beta, m_target):

                # zero gradients each run
                if m_gamma.grad is not None:
                    m_gamma.grad.zero_()
                if m_beta.grad is not None:
                    m_beta.grad.zero_()
                if m_x.grad is not None:
                    m_x.grad.zero_()

                # pass through custom layernorm operation (forward pass)
                mout = LayerNorm.apply(m_x, m_gamma, m_beta, 1e-5)

                # compute loss against target
                criterion = nn.MSELoss(reduction="sum")  # set mse sum (default is div)
                mloss = criterion(mout, m_target)

                # call backward pass via gradient outputs
                mloss.backward()
                
            manualLayerNorm(m_x, m_gamma, m_beta, m_target)
            """,
            globals={"m_x": m_x, "m_gamma": m_gamma, "m_beta": m_beta, "m_target": m_target}
        )
        
        # run iterations and caclulate runtime
        runtime = timer.adaptive_autorange(min_run_time=1)
        print("="*50)
        print("CASE", case, "\nRUNTIME", runtime)    
        print("="*50)

    



    
    
    
    
