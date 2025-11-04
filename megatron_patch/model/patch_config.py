from megatron.core.transformer import TransformerConfig
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class PatchedTransformerConfig(TransformerConfig):

    freeze_partial_moe_routers: Optional[bool] = False

    num_freezing_moe_routers: int = 0

    frozen_param_names: Optional[List[str]] = None

    n_extended_shared_experts: Optional[List[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.frozen_param_names is not None:
            if not self.freeze_partial_moe_routers and 'mlp.experts' in self.frozen_param_names:
                self.freeze_partial_moe_routers = True
                self.num_freezing_moe_routers = self.num_moe_experts
                self.moe_router_load_balancing_type = "none"
        if self.freeze_partial_moe_routers:
            assert self.num_freezing_moe_routers > 0
