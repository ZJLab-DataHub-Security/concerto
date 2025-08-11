# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.training.global_vars import get_args
from megatron.training.global_vars import get_signal_handler
from megatron.training.global_vars import get_tokenizer
from megatron.training.global_vars import get_tensorboard_writer
from megatron.training.global_vars import get_wandb_writer
from megatron.training.global_vars import get_one_logger
from megatron.training.global_vars import get_adlr_autoresume
from megatron.training.global_vars import get_timers
from megatron.training.initialize  import initialize_megatron
from megatron.training import pretrain, get_model, get_train_valid_test_num_samples

from megatron.training.utils import (print_rank_0,
                    is_last_rank,
                    print_rank_last)
