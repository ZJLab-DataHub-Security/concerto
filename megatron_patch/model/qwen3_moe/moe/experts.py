from megatron.core.transformer.moe.experts import GroupedMLP as _GroupedMLP
import torch
import torch.nn.functional as F
from functools import partial
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core import parallel_state as mpu
# from xmegatron_ext.core.transformer.moe.grouped_gemm_util import gmm_moefusion_impl as gmm
from xmegatron_ext.core.fusions.fused_bias_gelu import bias_gelu_impl
from xmegatron_ext.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from xmegatron_ext.core.pipeline_parallel.weight_grad_store import WeightGradStore
from torch.nn import Parameter
from typing import List

class MoeFcFusion_(torch.autograd.Function):
    """MoeFcFusion"""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        tokens_per_experts: torch.Tensor,
        trans_w: bool,
        weight: torch.nn.Parameter = None,
        weight_grad_offset: int = 0
    ):
        """forward"""
        from torch_xmlir._XMLIRC import (
            CachedBf16Initializer,
            bf16_status_initialize,
            initialized_group_linear,
            initialized_group_linear_weight_bwd,
        )
        ctx.weight_grad_offset = weight_grad_offset
        x_status_s, x_status_o = bf16_status_initialize(x)

        if isinstance(weight, Parameter) and not hasattr(weight, "hw_status_initializer"):
            setattr(weight, "hw_status_initializer", CachedBf16Initializer(inplace=True))
        ctx.weight_initializer = None
        if isinstance(weight, Parameter):
            assert hasattr(weight, "hw_status_initializer")
            ctx.weight_initializer = weight.hw_status_initializer

        if ctx.weight_initializer is not None:
            w_status_s, w_status_o = ctx.weight_initializer.initialize(w)
        else:
            w_status_s, w_status_o = bf16_status_initialize(w)

        # 构造lod，moe_fc_fusion需要lod在device上
        assert tokens_per_experts.ndim == 1, "tokens_per_experts must be 1-dim"
        pad = torch.zeros((tokens_per_experts.shape[0] + 1,), dtype=torch.int32, device=x.device)
        pad[1:].copy_(tokens_per_experts)
        lod = torch.cumsum(pad, dim=0).int()

        # print(f"MoeFcFusion {lod=}, {lod.device=}")

        assert w.ndim == 3, "w must be 3-dim"
        assert x.ndim == 2, "w must be 2-dim"
        assert x.dtype == torch.bfloat16, "x must be bfloat16"
        assert w.dtype == torch.bfloat16, "w must be bfloat16"

        if trans_w:
            output_shape = (x.shape[0], w.shape[1])
        else:
            output_shape = (x.shape[0], w.shape[2])

        out = torch.empty(output_shape, dtype=torch.bfloat16, device=x.device)

        initialized_group_linear(
            x_status_o,
            x.shape,
            w_status_o,
            w.shape,
            x_status_s,
            w_status_s,
            lod,
            out=out,
            other_trans=trans_w,
        )
        ctx.save_for_backward(weight, x_status_o, w_status_o, x_status_s, w_status_s, lod)
        ctx.x_shape = x.shape
        ctx.w_shape = w.shape
        ctx.trans_w = trans_w

        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """backward"""
        from torch_xmlir._XMLIRC import (
            CachedBf16Initializer,
            bf16_status_initialize,
            initialized_group_linear,
            initialized_group_linear_weight_bwd,
        )
        weight, x_status_o, w_status_o, x_status_s, w_status_s, lod = ctx.saved_tensors
        trans_w = ctx.trans_w
        x_shape = ctx.x_shape
        w_shape = ctx.w_shape

        assert dy.ndim == 2, "dy must be 2-dim"
        assert dy.dtype == torch.bfloat16, "dy must be bfloat16"
        assert dy.is_contiguous(), "dy must be contiguous"

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            dy_s, dy_o = bf16_status_initialize(dy)

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty(x_shape, dtype=torch.bfloat16, device=x_status_o.device)

            initialized_group_linear(
                dy_o,
                dy.shape,
                w_status_o,
                w_shape,
                dy_s,
                w_status_s,
                lod,
                out=dx,
                other_trans=not trans_w,
            )

        dw = None
        if ctx.needs_input_grad[1]:
            main_grad = weight.main_grad
            grad_output = torch.zeros_like(main_grad).view(w_shape)
            if trans_w:
                if WeightGradStore.enabled:
                    WeightGradStore.put(
                        initialized_group_linear_weight_bwd,
                        dy_o,
                        dy.shape,
                        x_status_o,
                        x_shape,
                        dy_s,
                        x_status_s,
                        lod,
                        beta=1.0,
                        self_trans=True,
                        # out=weight.main_grad.view(w_shape),
                        out=grad_output,
                    )
                else:
                    initialized_group_linear_weight_bwd(
                        dy_o,
                        dy.shape,
                        x_status_o,
                        x_shape,
                        dy_s,
                        x_status_s,
                        lod,
                        beta=1.0,
                        self_trans=True,
                        # out=weight.main_grad.view(w_shape),
                        out=grad_output,
                    )
            else:
                if WeightGradStore.enabled:
                    WeightGradStore.put(
                        initialized_group_linear_weight_bwd,
                        x_status_o,
                        x_shape,
                        dy_o,
                        dy.shape,
                        x_status_s,
                        dy_s,
                        lod,
                        beta=1.0,
                        self_trans=True,
                        # out=weight.main_grad.view(w_shape),
                        out=grad_output,
                    )
                else:
                    initialized_group_linear_weight_bwd(
                        x_status_o,
                        x_shape,
                        dy_o,
                        dy.shape,
                        x_status_s,
                        dy_s,
                        lod,
                        beta=1.0,
                        self_trans=True,
                        # out=weight.main_grad.view(w_shape),
                        out=grad_output,
                    )
            main_grad.view(w_shape)[ctx.weight_grad_offset:] = grad_output[ctx.weight_grad_offset:]
            # print(f"main_grad: {main_grad}")
            if hasattr(weight, "grad_added_to_main_grad"):
                if getattr(weight, "zero_out_wgrad", False):
                    dw = torch.zeros(
                        w_shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    dw = torch.empty(
                        w_shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                dw = None
        return dx, dw, None, None, None, None


def gmm(a, b, batch_sizes, trans_b=False, weight=None, freeze_weight_offset=0):
    """gmm_moefusion_impl"""
    return MoeFcFusion_.apply(a, b, batch_sizes, trans_b, weight, freeze_weight_offset)

class GroupedMLP(_GroupedMLP):
    def __init__(self, num_local_experts, config):
        super().__init__(num_local_experts, config)
        if self.config.bias_activation_fusion:
            if self.config.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    self.activation_func = partial(bias_gelu_impl, bias=None)
                else:
                    assert self.config.add_bias_linear is True
                    self.activation_func = partial(bias_gelu_impl, bias=None)
            elif self.config.activation_func == F.silu and self.config.gated_linear_unit:
                self.activation_func = partial(bias_swiglu_impl, bias=None)
            else:
                raise ValueError("Only support fusion of gelu and swiglu")

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        from megatron.core import tensor_parallel

        # from xmegatron_ext.core.transformer.moe.grouped_gemm_util import gmm_moefusion_impl as gmm

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)
        ep_rank = mpu.get_expert_model_parallel_rank()
        ep_frozen_rank = self.config.num_freezing_moe_routers // self.num_local_experts
        freeze_weight = False
        freeze_weight_offset = 0
        if ep_rank < ep_frozen_rank:
            freeze_weight = True
        if ep_rank == ep_frozen_rank:
            freeze_weight_offset = self.config.num_freezing_moe_routers % self.num_local_experts
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
            if freeze_weight:
                if not hasattr(self, 'frozen_w1'):
                    self.frozen_w1 = w1.detach()
                    self.frozen_w1.requires_grad = False
                if not hasattr(self, 'frozen_w2'):
                    self.frozen_w2 = w2.detach()
                    self.frozen_w2.requires_grad = False
                fc1_output = gmm(
                    permuted_local_hidden_states, self.frozen_w1, tokens_per_expert, trans_b=False, weight=self.weight1
                )
            else:
                fc1_output = gmm(
                    permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False, weight=self.weight1, freeze_weight_offset=freeze_weight_offset
                )
            if self.activation_recompute:
                intermediate_parallel = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, fc1_output, permuted_probs.unsqueeze(-1)
                )
                if freeze_weight:
                    fc2_output = gmm(
                        intermediate_parallel, self.frozen_w2, tokens_per_expert, trans_b=False, weight=self.weight2
                    )
                else:
                    fc2_output = gmm(
                        intermediate_parallel, w2, tokens_per_expert, trans_b=False, weight=self.weight2, freeze_weight_offset=freeze_weight_offset
                    )
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                if freeze_weight:
                    fc2_output = gmm(
                        intermediate_parallel, self.frozen_w2, tokens_per_expert, trans_b=False, weight=self.weight2
                    )
                else:
                    fc2_output = gmm(
                        intermediate_parallel, w2, tokens_per_expert, trans_b=False, weight=self.weight2, freeze_weight_offset=freeze_weight_offset
                    )
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                h = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None