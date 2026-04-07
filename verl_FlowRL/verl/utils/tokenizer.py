# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Utils for tokenization."""

import json
import os
import warnings

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1)
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer_kwargs = dict(kwargs)

    # Compatibility shim: older transformers builds expect extra_special_tokens
    # to be a dict, while some tokenizer_config.json files store it as a list.
    config_path = os.path.join(str(name_or_path), "tokenizer_config.json")
    if os.path.isfile(config_path) and "extra_special_tokens" not in tokenizer_kwargs:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_obj = json.load(f)
            extra_special_tokens = config_obj.get("extra_special_tokens")
            if isinstance(extra_special_tokens, list):
                tokenizer_kwargs["extra_special_tokens"] = {f"extra_special_token_{i}": tok for i, tok in enumerate(extra_special_tokens)}
                warnings.warn(
                    "Loaded tokenizer_config.json with list-valued extra_special_tokens; "
                    "converting to dict for transformers compatibility.",
                    stacklevel=1,
                )
        except Exception:
            # Fall through to normal loading if tokenizer config cannot be parsed.
            pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **tokenizer_kwargs)
    except AttributeError as exc:
        msg = str(exc)
        if "list" in msg and "keys" in msg and "extra_special_tokens" not in tokenizer_kwargs:
            retry_kwargs = dict(tokenizer_kwargs)
            retry_kwargs["extra_special_tokens"] = {}
            warnings.warn(
                "Retrying tokenizer load with empty extra_special_tokens due to "
                "transformers compatibility issue.",
                stacklevel=1,
            )
            tokenizer = AutoTokenizer.from_pretrained(name_or_path, **retry_kwargs)
        else:
            raise

    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        processor = None
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
