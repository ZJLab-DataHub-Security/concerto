# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Union
import argparse
import dataclasses
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig

def core_transformer_config_from_args(args, config_class=None):
    # Config class.
    config_class = config_class or TransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    #kw_args['first_pipeline_num_layers'] = args.decoder_first_pipeline_num_layers
    #kw_args['last_pipeline_num_layers'] = args.decoder_last_pipeline_num_layers
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion

    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None
    kw_args['config_logger_dir'] = args.config_logger_dir

    # Return config.
    return config_class(**kw_args)

def patch_if_not_exist(
        group_or_parser: Union[argparse._ArgumentGroup, argparse.ArgumentParser],
        keyname, type=None, default=None, help=None
):
    has_keyname = False
    for action in vars(group_or_parser)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if keyname in action.option_strings:
                has_keyname = True

    if not has_keyname:
        return group_or_parser.add_argument(
            keyname,
            type=type,
            default=default,
            help=help,
        )
    return None


def get_patch_args(parser):
    group = parser.add_argument_group(title="patch")

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--tokenizer-type" in action.option_strings:
                action.default = "NullTokenizer"

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--vocab-size" in action.option_strings:
                action.default = -1

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--optimizer" in action.option_strings:
                action.choices.append("hybridadam")

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--position-embedding-type" in action.option_strings:
                action.choices.append("none")

    has_rotary_base = False
    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--rotary-base" in action.option_strings:
                has_rotary_base = True

    patch_if_not_exist(
        group,
        "--rotary-base",
        type=int,
        default=10000,
        help="Base to use for rotary positional embeddings, default 10000",
    )

    patch_if_not_exist(
        group,
        "--local-rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher",
    )

    patch_if_not_exist(
        group,
        "--spatial-merge-size",
        type=int,
        default=2,
    )

    patch_if_not_exist(
        group,
        "--temporal-patch-size",
        type=int,
        default=2,
    )

    patch_if_not_exist(
        group,
        "--patch-size",
        type=int,
        default=14,
    )

    group.add_argument("--n-head-kv", type=int, default=None, help="n-head-kv")

    group.add_argument(
        "--transformer-type", type=str, default="megatron", help="transformer-type"
    )

    group.add_argument(
        "--max-padding-length", type=int, default=None, help="max-padding-length"
    )

    group.add_argument("--dataset", type=str, default=None, help="dataset")

    group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of finetunning epochs. Zero results in " "evaluation only.",
    )

    group.add_argument(
        "--intermediate-size", type=int, default=None, help="--intermediate-size"
    )

    group.add_argument(
        "--extra-vocab-size", type=int, default=0, help="--extra-vocab-size"
    )

    group.add_argument(
        "--keep-last",
        action="store_true",
        help="Keep the last batch (maybe incomplete) in" "the data loader",
    )

    group.add_argument("--data-dir", default=None, help="data-dir")

    group.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help="Whitespace separated paths or corpora names " "for training.",
    )

    group.add_argument(
        "--valid-data", nargs="+", default=None, help="path(s) to the validation data."
    )

    group.add_argument("--patch-tokenizer-type", type=str, help="patch-tokenizer-type")

    group.add_argument(
        "--use-alibi-mask",
        action="store_true",
        help="use alibi mask for baichuan model",
    )

    group.add_argument("--use-normhead", action="store_true", help="use-normhead")

    group.add_argument("--glu-activation", type=str, help="GLU activations to use.")

    group.add_argument(
        "--attention-head-type",
        type=str,
        default=None,
        choices=["multihead", "multiquery"],
        help="Type of attention heads. `multihead` is the standard multi-head attention."
        "`multiquery` shares the values and keys across attention heads",
    )

    group.add_argument(
        "--transformer-timers",
        action="store_true",
        help="If set, activate the timers within the transformer layers."
        "Only for debugging, as this slows down the model.",
    )

    group.add_argument("--text-generate-input-file", type=str, default="")

    group.add_argument("--text-generate-output-file", type=str, default="")

    group.add_argument("--text-generate-gt-file", type=str, default="")

    group.add_argument(
        "--time",
        action="store_true",
        help="measure end to end text generation average time",
    )

    group.add_argument("--eval-dev", action="store_true")

    group.add_argument(
        "--input-len",
        type=int,
        default=1,
        help="input lenth for measure end to end text generation average time",
    )

    group.add_argument(
        "--generation-length", type=int, default=None, help="generation-seq-len"
    )

    group.add_argument("--top-p", type=float, default=0.0, help="Top p sampling.")

    group.add_argument("--top-k", type=int, default=0, help="Top k sampling.")

    group.add_argument(
        "--out-seq-length",
        type=int,
        default=1024,
        help="Size of the output generated text.",
    )

    group.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )

    group.add_argument(
        "--repetition_penalty", type=float, default=1.1, help="Repetition_penalty."
    )

    group.add_argument(
        "--embed-layernorm", action="store_true", help="use layernorm for embedding"
    )

    group.add_argument(
        "--repetition-penalty", type=float, default=1.2, help="Repetition_penalty."
    )

    group.add_argument(
        "--source-seq-len", type=int, default=None, help="source-seq-len"
    )

    group.add_argument(
        "--target-seq-len", type=int, default=None, help="target-seq-len"
    )

    group.add_argument(
        "--position-encoding-2d", action="store_true", help="position-encoding-2d"
    )

    group.add_argument(
        "--z-loss-weight",
        type=float,
        default=0.0,
        help="the max-z weight for baichuan2",
    )

    group.add_argument(
        "--use-llama2-rotary-position-embeddings",
        action="store_true",
        help="Use llama2 rotary positional embeddings or not. "
        "Deprecated: use --position-embedding-type",
    )

    group.add_argument(
        "--use-mistral-rotary-position-embeddings",
        action="store_true",
        help="Use llama2 rotary positional embeddings or not. "
        "Deprecated: use --position-embedding-type",
    )

    group.add_argument("--mm-use-im-start-end", action="store_true")

    group.add_argument("--mm-use-im-patch-token", action="store_true")

    group.add_argument("--tune-mm-mlp-adapter", action="store_true")

    group.add_argument("--freeze-clip-vision-tower", action="store_true")

    group.add_argument("--freeze-llm", action="store_true")

    group.add_argument("--image-folder", type=str, default="")

    group.add_argument("--mm-vision-select-layer", type=int, default=None)

    group.add_argument("--vision-tower", type=str, default="")

    group.add_argument("--image-aspect-ratio", type=str, default="square")

    group.add_argument("--version", type=str, default="plain")

    group.add_argument("--mm-projector-type", type=str, default=None)

    group.add_argument("--image-size", type=int, default=None, help="image-size")


    group.add_argument("--sliding-window", type=int, default=None)

    group.add_argument("--rotary-scale-factor", type=int, default=1)

    group.add_argument("--cvcuda-image-processing", action="store_true")

    group.add_argument(
        "--expert-tensor-parallelism",
        action="store_true",
        default=False,
        help="use tensor parallelism for expert layers in MoE",
    )

    group.add_argument(
        "--expert-interval",
        type=int,
        default=2,
        help='Use experts in every "expert-interval" layers',
    )

    group.add_argument("--moe", action="store_true")

    group.add_argument("--moe-topk", type=int, default=1, help="moe-topk")

    group.add_argument(
        "--moe-expert-parallel-size",
        type=int,
        default=None,
        help="Degree of the MoE expert parallelism. By default, "
        "the size of this value will be automatically determined.",
    )

    group.add_argument(
        "--moe-train-capacity-factor",
        type=float,
        default=1.0,
        help="The capacity of the MoE expert at training time",
    )

    group.add_argument(
        "--moe-eval-capacity-factor",
        type=float,
        default=1.0,
        help="The capacity of the MoE expert at eval time.",
    )

    group.add_argument(
        "--moe-min-capacity",
        type=int,
        default=4,
        help="The minimum capacity per MoE expert regardless of the capacity_factor.",
    )

    group.add_argument(
        "--moe-loss-coeff",
        type=float,
        default=0.01,
        help="Scaling coefficient for adding MoE loss to model loss",
    )

    group.add_argument(
        "--use-tutel", action="store_true", help="Use Tutel optimization for MoE"
    )

    group.add_argument(
        "--router-type",
        type=str,
        default="topk",
        choices=["topk", "expert_choice"],
        help="Options for router type, support top1 & top2 and expert_choice",
    )

    group.add_argument(
        "--moe-input-feature-slicing",
        action="store_true",
        help="Enable moe all2all performance optimization.",
    )

    group.add_argument(
        "--disable-bias-linear-fc",
        action="store_false",
        help="Disable bias in the linear layers",
        dest="add_bias_linear_fc",
    )

    group.add_argument(
        "--disable-bias-attn-fc",
        action="store_false",
        help="Disable bias in the linear layers",
        dest="add_bias_attn_fc",
    )

    group.add_argument(
        "--disable-parallel-output",
        action="store_false",
        help="Disable parallel-output",
        dest="enable_parallel_output",
    )

    group.add_argument(
        "--task-list",
        type=str,
        default="all",
        help='Either "all" or comma separated list of tasks.',
    )

    group.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )

    group.add_argument(
        "--adaptive-seq-len",
        default=False,
        action="store_true",
        help="Should the sequence length be adapted to the batch during evaluation,"
        " if in fp16 the results will be slightly different due to numerical"
        " errors but greatly speed up evaluation.",
    )

    group.add_argument(
        "--eval-fp32",
        default=False,
        action="store_true",
        help="Should the evaluation run in fp32",
    )

    group.add_argument("--num-fewshot", type=int, default=None, help="num fewshot")

    group.add_argument(
        "--convert-checkpoint-from-megatron-to-transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    #group.add_argument("--moe-ffn-hidden-size", type=int, default=None)
    patch_if_not_exist(
        group,
        "--moe-ffn-hidden-size", type=int, default=None
    )

    group.add_argument("--shared-moe-ffn-hidden-size", type=int, default=None)

    group.add_argument(
        "--enable-shared-expert", action="store_true", help="enable-shared-expert"
    )

    patch_if_not_exist(
        group,
        "--q-lora-rank", type=int, default=None
    )

    patch_if_not_exist(
        group,
        "--kv-lora-rank", type=int, default=None
    )

    patch_if_not_exist(
        group,
        "--v-head-dim", type=int, default=None
    )

    group.add_argument("--qk-nope-head-dim", type=int, default=None)
    group.add_argument("--qk-rope-head-dim", type=int, default=None)
    group.add_argument("--num-shared-experts", type=int, default=None)
    #group.add_argument("--moe-layer-freq", type=int, default=1)
    patch_if_not_exist(
        group,
        "--moe-layer-freq", type=int, default=1
    )

    patch_if_not_exist(
        group,
        "--rotary-scaling-factor", type=int, default=1
    )

    group.add_argument(
        "--optimizer-offload-policy",
        default="static",
        type=str,
        help="Optimizer Offload Policy used by OffloadDistributedOptimizer, "
        "valid if base optimizer is HybridAdam.",
    )

    group.add_argument(
        "--train-mode", default="pretrain", type=str, help="pretrain or finetune"
    )

    group.add_argument(
        "--optimizer-offload-auto-threshold",
        type=int,
        default=2048 * 1024 * 1024,
        help="Optimizer Offload Threshold currently used by auto policy, "
        "tune larger if OOM occurs",
    )

    group.add_argument(
        "--optimizer-offload-chunk-size",
        type=int,
        default=32 * 1024 * 1024,
        help="Chunk size of Chunk Manager in Optimizer Offload,"
        "keep zero to search for a optimal size",
    )

    group.add_argument(
        "--cpu-offloading",
        default=False,
        action="store_true",
        help="Use activation checkpointing.",
    )

    group.add_argument(
        "--cpu-offloading-num-layers",
        type=int,
        default=0,
        help="The num of layers to be moved to CPU",
    )

    group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    group.add_argument('--freeze-LM', action='store_true', default=False)
    group.add_argument('--freeze-ViT', action='store_true', default=False)
    group.add_argument('--language-model-type', type=str, required=False)
    group.add_argument('--vision-model-type', type=str, default="clip")
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint", action="store_true", default=False
    )
    group.add_argument("--use-te", action="store_true", default=False)
    group.add_argument(
        "--dataloader-save", type=str, default=None, help="Energon dataloader state save path"
    )
    group.add_argument(
        "--use-tiling", action="store_true", default=False, help="Use input image tiling"
    )
    group.add_argument("--max-num-tiles", type=int, default=1, help="Maximum number of image tiles")
    group.add_argument(
        "--use-thumbnail", action="store_true", default=False, help="Add image thumbnail as a tile"
    )
    group.add_argument(
        "--dataloader-seq-length",
        type=int,
        help="Make dataloader to produce sequences of specific length.",
    )
    group.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to regularly sample from the video as input to the model.",
    )
    group.add_argument(
        "--online-evaluation-config", type=str, help="Config file for online evaluation."
    )

    group.add_argument(
        "--tokenizer-prompt-format",
        type=str,
        choices=["mistral", "llama3", "chatml"],
        required=False,
        help="Prompt format to use with the tokenizer.",
    )

    group.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<image>"],
        help="Special tokens used in the multimodal model",
    )

    group.add_argument(
        "--image-tag-type",
        type=str,
        choices=["nvlm", "internvl", ""],
        default="",  # Default: Image tag not used.
        help="Surround image tokens with tags.",
    )
    group.add_argument('--log-mfu',
                    action='store_true',
                    help='If set, calculate and log MFU.')
    group.add_argument('--log-mfu-to-tensorboard',
                    action='store_true',
                    help='Enable MFU logging to tensorboard.')
    group.add_argument('--mfu-base-value', type=int, default=312,
                    help='The denominator when calculating the MFU. Default 312')
    group.add_argument('--online-packing',
                    action='store_true',
                    help='If set, dataloader tokenize and packing datasets online')
    group.add_argument('--te-spec-version',
                    type=str,
                    choices=["base", "tqlm"],
                    default="base",
                    help="support base or tqlm, base uses mcore-0.10, "
                    "tqlm uses mcore-0.11 and fuses input_layernorm and linear_qkv")
    return parser
