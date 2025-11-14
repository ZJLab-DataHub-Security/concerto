from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.transformer.moe.moe_layer import MoESubmodules, MoELayer as _MoELayer
from typing import Optional
from megatron.core.transformer.spec_utils import build_module
class MoELayer(_MoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)
        if self.use_shared_expert and self.config.n_extended_shared_experts is not None and layer_number <= len(self.config.n_extended_shared_experts):
            import copy
            new_config = copy.deepcopy(self.config)
            new_config.moe_shared_expert_intermediate_size = new_config.moe_shared_expert_intermediate_size * self.config.n_extended_shared_experts[layer_number-1]
            print(f"layer{layer_number} moe_shared_expert_intermediate_size={new_config.moe_shared_expert_intermediate_size}")
            if new_config.moe_shared_expert_intermediate_size > 0:
                self.shared_experts = build_module(self.submodules.shared_experts, config=new_config)
                if self.shared_expert_overlap:
                    self.token_dispatcher.set_shared_experts(self.shared_experts)
            else:
                self.shared_experts = None
                self.use_shared_expert = False
