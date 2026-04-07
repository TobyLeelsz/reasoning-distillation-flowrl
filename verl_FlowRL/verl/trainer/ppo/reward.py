# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import multiprocessing
import os
import hashlib
import json
from functools import partial

import numpy as np
import ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score

PROMPT_TOKEN_IDS_KEY = "__verl_prompt_token_ids"
RESPONSE_TOKEN_IDS_KEY = "__verl_response_token_ids"
PROMPT_TEXT_KEY = "__verl_prompt_text"
RESPONSE_TEXT_KEY = "__verl_response_text"
_CUSTOM_REWARD_FN_CACHE = {}


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    function_name = reward_fn_config.get("name")
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    try:
        kwargs_key = json.dumps(reward_kwargs, sort_keys=True, default=str)
    except TypeError:
        kwargs_key = repr(sorted(reward_kwargs.items()))
    cache_key = (os.path.realpath(file_path), function_name, kwargs_key)
    if cache_key in _CUSTOM_REWARD_FN_CACHE:
        return _CUSTOM_REWARD_FN_CACHE[cache_key]

    module_name = f"custom_module_{hashlib.md5(os.path.realpath(file_path).encode('utf-8')).hexdigest()}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    _CUSTOM_REWARD_FN_CACHE[cache_key] = wrapped_fn
    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "naive_code":
        from verl.workers.reward_manager import NaiveCodeRewardManager

        reward_manager_cls = NaiveCodeRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        reward_manager_cls = BatchRewardManager
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        reward_manager_cls = DAPORewardManager
    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        else:
            final_compute_score = default_compute_score

    reward_manager = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )
    # Expose whether we are using a custom reward function so we can attach
    # extra context (e.g., prompt/response token IDs) only when needed.
    reward_manager.uses_custom_reward_fn = compute_score is not None
    return reward_manager


def _attach_prompt_response_token_ids_to_extra_info(data: DataProto, tokenizer=None):
    if "prompts" not in data.batch or "responses" not in data.batch or "attention_mask" not in data.batch:
        return

    prompt_ids = data.batch["prompts"]
    response_ids = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]

    prompt_len = prompt_ids.shape[-1]
    valid_prompt_lengths = attention_mask[:, :prompt_len].sum(dim=-1)
    valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

    existing_extra_infos = data.non_tensor_batch.get("extra_info", None)
    if existing_extra_infos is None or len(existing_extra_infos) != len(data):
        existing_extra_infos = [None] * len(data)

    merged_extra_infos = []
    for i in range(len(data)):
        raw_extra_info = existing_extra_infos[i]
        merged_extra_info = dict(raw_extra_info) if isinstance(raw_extra_info, dict) else {}

        valid_prompt_len = int(valid_prompt_lengths[i].item())
        valid_response_len = int(valid_response_lengths[i].item())

        valid_prompt_token_ids = prompt_ids[i][-valid_prompt_len:].detach().cpu().tolist()
        valid_response_token_ids = response_ids[i][:valid_response_len].detach().cpu().tolist()

        merged_extra_info[PROMPT_TOKEN_IDS_KEY] = valid_prompt_token_ids
        merged_extra_info[RESPONSE_TOKEN_IDS_KEY] = valid_response_token_ids

        if tokenizer is not None:
            try:
                merged_extra_info[PROMPT_TEXT_KEY] = tokenizer.decode(
                    valid_prompt_token_ids,
                    skip_special_tokens=False,
                )
            except Exception:
                pass
            try:
                merged_extra_info[RESPONSE_TEXT_KEY] = tokenizer.decode(
                    valid_response_token_ids,
                    skip_special_tokens=False,
                )
            except Exception:
                pass

        merged_extra_infos.append(merged_extra_info)

    data.non_tensor_batch["extra_info"] = np.array(merged_extra_infos, dtype=object)


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    if getattr(reward_fn, "uses_custom_reward_fn", False):
        _attach_prompt_response_token_ids_to_extra_info(data, tokenizer=getattr(reward_fn, "tokenizer", None))

    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn (no retry): {e}")
        raise

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)


@ray.remote(num_cpus=1)
class AsyncRewardWorker:
    """Persistent async reward worker that keeps custom reward models warm."""

    def __init__(self, config, tokenizer):
        self.reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        try:
            print(f"[async_reward] initialized worker on gpu_ids={ray.get_gpu_ids()}")
        except Exception:
            pass

    def warmup(self):
        # Warmup custom reward to materialize heavyweight models before rollout.
        if not hasattr(self.reward_fn, "compute_score"):
            return {"status": "skipped", "reason": "no_compute_score"}
        result = self.reward_fn.compute_score(
            data_sources=np.array(["warmup"], dtype=object),
            solution_strs=["warmup"],
            ground_truths=[None],
            extra_infos=[None],
        )
        return {"status": "ok", "gpu_ids": ray.get_gpu_ids(), "result_len": len(result)}

    def compute(self, data: DataProto):
        return compute_reward(data, self.reward_fn)
