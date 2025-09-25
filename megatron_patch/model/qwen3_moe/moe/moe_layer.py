from megatron.core.transformer.transformer_config import TransformerConfig

from .router import TopKRouter
from megatron.core.transformer.moe.moe_layer import MoESubmodules, MoELayer as _MoELayer
from megatron.core.transformer.moe.moe_utils import ModelCommProcessGroups, get_default_model_comm_pgs
from typing import Optional

class MoELayer(_MoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        if model_comm_pgs is None:
            model_comm_pgs = get_default_model_comm_pgs()
        super(MoELayer, self).__init__(config=config, submodules=submodules, layer_number=layer_number, model_comm_pgs=model_comm_pgs)
        self.router = TopKRouter(config=self.config, model_comm_pgs=model_comm_pgs)