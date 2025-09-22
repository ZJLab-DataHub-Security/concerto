from megatron.core.transformer import TransformerConfig
from typing import Optional
from dataclasses import dataclass

@dataclass
class PatchedTransformerConfig(TransformerConfig):

    freeze_moe_router: Optional[bool] = False

    freeze_partial_moe_routers: Optional[bool] = False

    num_freezing_moe_routers: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.freeze_partial_moe_routers:
            self.freeze_moe_router = False
            assert self.num_freezing_moe_routers > 0 and self.num_freezing_moe_routers < self.num_moe_experts
        if self.freeze_moe_router:
            self.moe_router_load_balancing_type = "none"

