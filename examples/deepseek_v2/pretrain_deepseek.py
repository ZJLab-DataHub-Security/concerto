# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
from functools import partial
from typing import Union

import torch
import torch._dynamo

from xmegatron_ext import megatron_xpu_init
megatron_xpu_init(use_version="0.10.0")

from megatron.core.datasets.gpt_dataset import (
    GPTDatasetConfig,
)
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain, print_rank_0

from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer

from megatron.training.arguments import core_transformer_config_from_args
from megatron_patch.model.deepseek_v2.transformer_config import DeepSeekV2TransformerConfig
from megatron_patch.model.deepseek_v2.model import GPTModel
from megatron_patch.model.deepseek_v2.layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)

from megatron.core.transformer.transformer_config import MLATransformerConfig
MLATransformerConfig.original_max_position_embeddings = 4096

from megatron.core import parallel_state
parallel_state.get_data_modulo_expert_parallel_rank = lambda *args, **kwargs: torch.distributed.get_rank(group=parallel_state.get_expert_data_parallel_group())


#from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
#from megatron.core.models.gpt import GPTModel
from megatron_patch.data import train_valid_test_datasets_provider
from megatron_patch.tokenizer import build_tokenizer

torch._dynamo.config.suppress_errors = True

def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModel]:

    args = get_args()
    build_tokenizer(args)
    config = core_transformer_config_from_args(args, DeepSeekV2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"

    if use_te:
        print_rank_0("building deepseek_v2 model in TE...")
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, multi_latent_attention=True, fp8=args.fp8
        )
    else:
        raise ValueError("Current only support TE")

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    return model

def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path),
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


if __name__ == "__main__":    
    from megatron_patch.template.helper import forward_step
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
