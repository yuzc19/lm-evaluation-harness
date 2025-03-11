# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import importlib
import pathlib
from copy import deepcopy
from typing import List, Literal

import filelock
import numpy as np
import torch
from tqdm import tqdm
from enum import Enum, auto

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
    simple_parse_args_string,
)

import os
import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.training.checkpointing import get_rng_state, load_checkpoint, _load_base_checkpoint
from megatron.training.utils import print_rank_0

from megatron.training import get_args, get_tokenizer
from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.initialize import (
    setup_logging,
    _set_random_seed,
    _initialize_distributed,
    _init_autoresume,
    _initialize_tp_communicators,
    _compile_dependencies,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.inference.text_generation.api import generate_and_post_process


@register_model("megatron_lm")
class MegatronLM(LM):
    def __init__(
        self,
        path: str,
        tokenizer_model: str,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        devices: int = 1,
        num_nodes: int = 1,
        **kwargs,
    ):

        super().__init__()

        args = parse_args(ignore_unknown_args=True)

        args.load = path
        args.micro_batch_size = int(batch_size)
        args.tokenizer_model = tokenizer_model

        initialize_megatron_with_load_args(args)
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
        )
        load_checkpoint([model], None, None)
        model.float()

        self.args = args
        self.model = model

        self._max_length = self.args.max_position_embeddings
        self._batch_size = int(batch_size)
        self._max_gen_toks = max_gen_toks

        self.tokenizer = get_tokenizer()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        args = simple_parse_args_string(arg_string)
        if additional_config:
            args["batch_size"] = additional_config.get("batch_size", 1)

        return cls(**args)

    @property
    def eot_token_id(self):
        try:
            return self.args.eos_id
        except AttributeError:
            return None

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self.args.device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def accelerator(self):
        return self._Accelerator(self.world_size)

    class _Accelerator:
        def __init__(self, world_size):
            self.world_size = world_size

        def wait_for_everyone(self):
            torch.distributed.barrier()

        def gather(self, local_tensor):
            gathered_tensors = [
                torch.zeros(1, dtype=local_tensor.dtype).cuda()
                for _ in range(self.world_size)
            ]
            torch.distributed.all_gather(gathered_tensors, local_tensor)
            return torch.cat(gathered_tensors)

    def tok_encode(self, string: str):
        return self.tokenizer.tokenize(string)

    def tok_decode(self, tokens):
        return self.tokenizer.detokenize(tokens)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            ctxlens = []
            contlens = []

            for _, context_enc, continuation_enc in chunk:
                # Leave one token for generation. Tokens_to_generate = 0 breaks NeMo.
                inp = (context_enc + continuation_enc)[-(self.max_length) :]

                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )
                ctxlens.append(ctxlen)
                contlens.append(len(continuation_enc))

                inps.append(self.tok_decode(inp))

            # output = self.generate(
            #     self.model,
            #     inputs=inps,
            #     tokens_to_generate=1,
            #     min_tokens_to_generate=1,
            #     compute_logprob=True,
            #     all_probs=True,
            # )
            (
                batch_output,
                batch_tokens,
                batch_logprobs,
                batch_token_ids,
                batch_full_logprob,
            ) = generate_and_post_process(
                model=self.model,
                prompts=inps,
                tokens_to_generate=0,
                return_output_log_probs=True,
                return_logits=True,
            )

            # batch_token_ids = np.asarray(output["token_ids"])[:, :-1]
            # batch_logprobs = output["logprob"][:, :-1]
            # batch_full_logprob = output["full_logprob"][:, :-1, :]
            batch_token_ids = np.asarray(batch_token_ids)

            # Compute greedy tokens for entire batch rather than calling it with proper ctxlen for each sample.
            # Additional tokens for each sample will be trimmed later.
            min_ctxlen = min(ctxlens)

            # Use min_ctxlen-1 instead of min_ctxlen since full_logprobs are not returns for the first token.
            batch_greedy_tokens = (
                torch.argmax(batch_full_logprob[:, min_ctxlen - 1 :, :], -1)
                .cpu()
                .numpy()
            )

            for (
                token_ids,
                greedy_tokens,
                logprobs,
                ctxlen,
                contlen,
                (
                    cache_key,
                    _,
                    _,
                ),
            ) in zip(
                batch_token_ids,
                batch_greedy_tokens,
                batch_logprobs,
                ctxlens,
                contlens,
                chunk,
            ):
                # Trim at contlen since shorter contexts in a batch will have more than one token generated.
                # Use ctxlen-1 instead of ctxlen same as for full_logprob in batch_greedy_tokens calculation
                logprobs = (logprobs[ctxlen - 1 :])[:contlen]
                logprob = sum(logprobs)

                continuation_tokens = (token_ids[ctxlen:])[:contlen]
                len_diff = ctxlen - min_ctxlen
                is_greedy = continuation_tokens == (greedy_tokens[len_diff:])[:contlen]
                if not isinstance(is_greedy, bool):
                    is_greedy = is_greedy.all()
                answer = (logprob, is_greedy)

                if cache_key is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)
                pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(self, requests):
        assert False, "Not implemented"
        if not requests:
            return []
        res = []

        def get_until(req_args):
            until = req_args.get("until", [])
            until = deepcopy(until)  # prevent from modifying req_args for cache_key
            if self.tokenizer.ids_to_tokens([self.eot_token_id])[0] not in until:
                until.append(self.tokenizer.ids_to_tokens([self.eot_token_id])[0])
            return until

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ords = Collator(
            [reg.args for reg in requests], sort_fn=_collate, group_by="gen_kwargs"
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            req_args = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = get_until(req_args)
            max_gen_toks = req_args.get("max_gen_toks", self.max_gen_toks)

            remaining_length = self.max_length - max_gen_toks
            contexts = []
            for context, _ in chunk:
                encoded_context = self.tok_encode(context)
                encoded_context = encoded_context[-remaining_length:]
                contexts.append(self.tok_decode(encoded_context))

            output = self.generate(
                self.model,
                inputs=contexts,
                tokens_to_generate=max_gen_toks,
                end_strings=until,
                greedy=True,
            )

            answers = output["sentences"]

            continuations = []
            for context, answer in zip(contexts, answers):
                continuations.append(answer[len(context) :])

            for term in until:
                continuations = [answer.split(term)[0] for answer in continuations]

            for request, answer in zip(chunk, continuations):
                self.cache_hook.add_partial("greedy_until", request, answer)
                res.append(answer)

        return re_ords.get_original(res)


def initialize_megatron_with_load_args(
    args,
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """

    load_args_from_checkpoint(args)

    args.expert_model_parallel_size = 1

    validate_args(args)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # set logging level
    setup_logging()

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None


class CheckpointType(Enum):
    LEGACY = auto()
    LOCAL = auto()
    GLOBAL = auto()


def load_dist(load_dir, args):
    load_kwargs = {}
    is_dist_ckpt = False
    if (
        args.auto_detect_ckpt_format
        or args.use_dist_ckpt
        or args.non_persistent_save_interval is not None
    ):
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir,
            args,
            rank0=True,
        )
        if args.enable_ft_package and ft_client is not None and state_dict is not None:
            if "ft_state" in state_dict:
                ft_client.load_state_dict(state_dict["ft_state"])
            else:
                print_rank_0("ft_state is not present in state_dict")
        is_dist_ckpt = (
            ckpt_type == CheckpointType.LOCAL
            or dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
        )
        if is_dist_ckpt:
            ckpt_tp_pp = (
                state_dict["args"].tensor_model_parallel_size,
                state_dict["args"].pipeline_model_parallel_size,
                getattr(state_dict["args"], "encoder_tensor_model_parallel_size", 0),
                getattr(state_dict["args"], "encoder_pipeline_model_parallel_size", 0),
            )
            run_tp_pp = (
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                # TODO: change this to args.encoder_tensor_model_parallel_size after 30th Nov 24
                getattr(args, "encoder_tensor_model_parallel_size", 0),
                getattr(args, "encoder_pipeline_model_parallel_size", 0),
            )
            mismatch_msg = "(TP, PP, encoder TP, encoder PP) mismatch after resume ({} vs {} from checkpoint)".format(
                run_tp_pp, ckpt_tp_pp
            )

            # Determine if RNG state will be loaded
            if (
                ckpt_tp_pp == run_tp_pp
                and not release
                and not args.finetune
                and not args.no_load_rng
                and not getattr(state_dict["args"], "no_save_rng", False)
            ):
                gen_sd_rng_state = get_rng_state(True)  # we can load the rng state
            else:
                gen_sd_rng_state = None
                if ckpt_tp_pp != run_tp_pp:
                    print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

            optim_sd_kwargs = dict(is_loading=True)
            # Determine if optimizer state will be loaded
            if (
                not release
                and not args.finetune
                and not args.no_load_optim
                and not getattr(state_dict["args"], "no_save_optim", False)
            ):
                gen_sd_optim = optimizer
                gen_sd_opt_param_scheduler = opt_param_scheduler

                if args.use_distributed_optimizer:
                    optim_sd_kwargs["sharding_type"] = (
                        "fully_sharded_model_space"
                        if getattr(
                            state_dict["args"], "ckpt_fully_parallel_save", False
                        )
                        else "dp_zero_gather_scatter"
                    )
                    # This is for backwards-compatibility. Can be removed once 'fully_sharded_bucket_space' loading is removed
                    for maybe_dist_opt_optim_state in (
                        state_dict["optimizer"],
                        *state_dict["optimizer"].values(),
                    ):
                        if "param_state_sharding_type" in maybe_dist_opt_optim_state:
                            if (
                                maybe_dist_opt_optim_state["param_state_sharding_type"]
                                == "fully_sharded_bucket_space"
                            ):
                                print_rank_0(
                                    "Detected deprecated `fully_sharded_bucket_space` DistributedOptimizer checkpoint format"
                                )
                                optim_sd_kwargs["sharding_type"] = (
                                    maybe_dist_opt_optim_state[
                                        "param_state_sharding_type"
                                    ]
                                )
                            break

                    if (
                        ckpt_tp_pp != run_tp_pp
                        and optim_sd_kwargs["sharding_type"]
                        != "fully_sharded_model_space"
                    ):
                        raise RuntimeError(
                            f"{mismatch_msg}: not supported for DistributedOptimizer with sharding type {optim_sd_kwargs['sharding_type']}."
                            f" Please use `--ckpt-fully-parallel-save` flag during checkpoint saving."
                        )
            else:
                gen_sd_optim = None
                gen_sd_opt_param_scheduler = None

            # [ModelOpt]: Initial loading from non-resume sharded checkpoint to a Distillation Model
            # will result in key mismatch with loss modules potentially containing parameters, since
            # it requires generating a state_dict before loading. Here we hide those modules if present.
            with contextlib.ExitStack() as stack:  # Allows multiple context managers for each model shard
                if args.finetune and hasattr(model[0], "hide_loss_modules"):
                    for m in model:
                        stack.enter_context(m.hide_loss_modules())
                load_kwargs["sharded_state_dict"] = generate_state_dict(
                    args,
                    model,
                    gen_sd_optim,
                    gen_sd_opt_param_scheduler,
                    gen_sd_rng_state,
                    use_dist_ckpt=True,
                    optim_sd_kwargs=optim_sd_kwargs,
                    train_data_iterator=None,
                )

            # When "--fp8-param-gather" is disabled, this function doesn't modify anything.
            fix_fp8_params_lose_precision_when_loading_dist_ckpt(
                load_kwargs["sharded_state_dict"]
            )


def load_args_from_checkpoint(args, load_arg="load"):
    """Set required arguments from the checkpoint specified in the
    arguments.

    Will overwrite arguments that have a non-None default value, but
    will leave any arguments that default to None as set.

    Returns the same args NameSpace with the new values added/updated.

    If no checkpoint is specified in args, or if the checkpoint is
    there but invalid, the arguments will not be modified

    """
    load_dir = getattr(args, load_arg)

    if load_dir is None:
        print_rank_0("No load directory specified, using provided arguments.")
        return args

    # args.use_dist_ckpt = True
    # load_dist(load_dir, args)
    state_dict, checkpoint_name, release, _ = _load_base_checkpoint(
        load_dir, args, rank0=True
    )
    print(state_dict)

    # Args.
    if not state_dict:
        print_rank_0(
            "Checkpoint not found to provide arguments, using provided arguments."
        )
        return args

    if "args" not in state_dict:
        print_rank_0(
            "Checkpoint provided does not have arguments saved, using provided arguments."
        )
        return args

    checkpoint_args = state_dict["args"]
    checkpoint_version = state_dict.get("checkpoint_version", 0)
    args.iteration = state_dict["iteration"]

    # One-off conversion for foundation models
    if hasattr(checkpoint_args, "disable_bias_linear"):
        setattr(
            checkpoint_args,
            "add_bias_linear",
            not getattr(checkpoint_args, "disable_bias_linear"),
        )

    def _set_arg(arg_name, old_arg_name=None, force=True):
        if not force and getattr(args, arg_name, None) is not None:
            print_rank_0(
                f"Argument {arg_name} already set to {getattr(args, arg_name)}"
            )
            return

        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name, None)

        if checkpoint_value is not None:
            print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
            setattr(args, arg_name, checkpoint_value)
        else:
            print_rank_0(f"Checkpoint did not provide arguments {arg_name}")

    _set_arg("num_layers")
    _set_arg("hidden_size")
    _set_arg("ffn_hidden_size")
    _set_arg("seq_length")
    _set_arg("num_attention_heads")
    _set_arg("num_query_groups", force=True)
    _set_arg("group_query_attention", force=True)
    _set_arg("kv_channels")
    _set_arg("max_position_embeddings")
    _set_arg("position_embedding_type", force=True)
    _set_arg("add_position_embedding", force=True)
    _set_arg("use_rotary_position_embeddings", force=True)
    _set_arg("rotary_percent", force=True)
    _set_arg("rotary_interleaved", force=True)
    _set_arg("add_bias_linear", force=True)
    _set_arg("add_qkv_bias", force=True)
    _set_arg("swiglu", force=True)
    _set_arg("untie_embeddings_and_output_weights", force=True)
    _set_arg("apply_layernorm_1p", force=True)
    _set_arg("normalization", force=True)
    _set_arg("tokenizer_type")
    _set_arg("padded_vocab_size")
    _set_arg("apply_query_key_layer_scaling", force=True)
    if checkpoint_version < 3.0:
        _set_arg("tensor_model_parallel_size", "model_parallel_size")
    else:
        _set_arg("tensor_model_parallel_size", force=True)
        _set_arg("pipeline_model_parallel_size", force=True)
        _set_arg("virtual_pipeline_model_parallel_size", force=True)
        _set_arg("num_layers_per_virtual_pipeline_stage")
    _set_arg("non_persistent_global_ckpt_dir")
    _set_arg("non_persistent_ckpt_type")
    _set_arg("no_persist_layer_norm")
    _set_arg("norm_epsilon")
    _set_arg("params_dtype")
    _set_arg("overlap_p2p_comm")

    _set_arg("num_experts")
    _set_arg("bias_swiglu_fusion")
    _set_arg("squared_relu")
    _set_arg("init_method_xavier_uniform")
    _set_arg("config_logger_dir")
    _set_arg("rotary_base")
    _set_arg("use_flash_attn")
    _set_arg("use_legacy_models")
    _set_arg("make_vocab_size_divisible_by")

    args.max_tokens_to_oom = 10000000

    # set all the arguments in TransformerConfig
    for key in TransformerConfig.__dataclass_fields__.keys():
        _set_arg(key, force=True)

    return args, checkpoint_args
