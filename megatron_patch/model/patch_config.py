from megatron.core.transformer import TransformerConfig
from typing import Optional
from dataclasses import dataclass

@dataclass
class PatchedTransformerConfig(TransformerConfig):

    freeze_moe_router: Optional[bool] = False

    def __post_init__(self):
        super().__post_init__()
        if self.freeze_moe_router:
            self.moe_router_load_balancing_type = "none"
