import torch
from megatron.core.transformer.moe.router import TopKRouter as _TopKRuter
from megatron.core.transformer.moe.moe_utils import router_gating_linear, te_general_gemm

class RouterGatingLinearFunction(torch.autograd.Function):
    """
    Autograd function for router gating linear.
    """

    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor, router_dtype: torch.dtype, requires_weight_grad: bool):
        """
        Forward pass of the RouterGatingLinearFunction function.
        """
        ctx.save_for_backward(inp, weight)
        ctx.router_dtype = router_dtype
        ctx.input_dtype = inp.dtype
        ctx.weight_dtype = weight.dtype
        ctx.requires_weight_grad = requires_weight_grad
        inp_shape = inp.shape
        inp = inp.view(-1, inp_shape[-1])

        if te_general_gemm is not None and router_dtype != torch.float64:
            output = te_general_gemm(weight, inp, router_dtype, layout="TN")
            output = output[0]
        else:
            output = torch.mm(inp.to(router_dtype), weight.to(router_dtype).t())

        output = output.view(*inp_shape[:-1], -1)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass of the RouterGatingLinearFunction function.
        """
        inp, weight = ctx.saved_tensors
        inp_shape = inp.shape
        grad_shape = grad_output.shape
        inp = inp.view(-1, inp_shape[-1])
        grad_output = grad_output.view(-1, grad_shape[-1])

        if te_general_gemm is not None and ctx.router_dtype != torch.float64:
            grad_input = te_general_gemm(
                weight.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NN", grad=True
            )
            grad_input = grad_input[0].to(ctx.input_dtype)
            if ctx.requires_weight_grad:
                grad_weight = te_general_gemm(
                    inp.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NT", grad=True
                )
                grad_weight = grad_weight[0].to(ctx.weight_dtype)
            else:
                grad_weight = None
        else:
            grad_input = torch.mm(grad_output, weight.to(ctx.router_dtype)).to(ctx.input_dtype)
            if ctx.requires_weight_grad:
                grad_weight = torch.mm(grad_output.t(), inp.to(ctx.router_dtype)).to(ctx.weight_dtype)
            else:
                grad_weight = None
        grad_input = grad_input.view(*inp_shape)
        return grad_input, grad_weight, None, None


def _router_gating_linear(inp: torch.Tensor, weight: torch.Tensor, router_dtype: torch.dtype, requires_weight_grad: bool):
    """
    Customized linear layer for router gating.
    This linear layer accepts bfloat16 input and weight, and can return output with router_dtype.
    It can reduce the memory usage by avoiding saving the intermediate high precision tensors.
    """
    return RouterGatingLinearFunction.apply(inp, weight, router_dtype, requires_weight_grad)

class TopKRouter(_TopKRuter):
    """Route each token to the top-k experts."""
    def __init__(self, config, model_comm_pgs):
        super().__init__(config=config, model_comm_pgs=model_comm_pgs)
        if hasattr(self.config, "freeze_moe_router") and self.config.freeze_moe_router:
            self.weight.requires_grad = False
        if hasattr(self.config, "freeze_partial_moe_routers") and self.config.freeze_partial_moe_routers:
            self.frozen_weight = self.weight[:self.config.num_freezing_moe_routers].detach()
            self.active_weight = self.weight[self.config.num_freezing_moe_routers:]
            self.frozen_weight.requires_grad = False
    
    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if hasattr(self.config, "freeze_partial_moe_routers") and self.config.freeze_partial_moe_routers:
            self.frozen_weight.data = self.frozen_weight.data.to(device=torch.cuda.current_device())
            self.active_weight.data = self.active_weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        if not self.config.freeze_partial_moe_routers:
            logits = router_gating_linear(input, self.weight, router_dtype)
        else:
            logits_1 = _router_gating_linear(input, self.frozen_weight, router_dtype, requires_weight_grad=False)
            logits_2 = _router_gating_linear(input, self.active_weight, router_dtype, requires_weight_grad=True)
            logits = torch.concat([logits_1, logits_2], dim=-1)
        return logits