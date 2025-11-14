from megatron.core.transformer import TransformerConfig
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class PatchedTransformerConfig(TransformerConfig):

    frozen_param_names: Optional[List[str]] = None

    n_extended_shared_experts: Optional[List[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.frozen_param_names is not None:
            if 'mlp.router' in self.frozen_param_names:
                self.moe_router_load_balancing_type = "none"

