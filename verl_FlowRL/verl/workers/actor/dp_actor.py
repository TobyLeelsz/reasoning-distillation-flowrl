# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False,return_log_z=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    output_hidden_states=True if return_log_z else False, # for estimate log z
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            if return_log_z:
                last_hidden = output.hidden_states[-1].squeeze(0) # (total_nnz, hidden size)
                if self.use_ulysses_sp:
                        last_hidden = gather_outpus_and_unpad(
                            last_hidden,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size, 
                        )
                full_last_hidden = pad_input(hidden_states=last_hidden,
                                        indices=indices,
                                        batch=batch_size,
                                        seqlen=seqlen)
                # extract pormpt hiddenstate for log z
                prompts_last_hidden = full_last_hidden[:, : -response_length - 1]
                prompt_attention_mask = attention_mask[:, : -response_length - 1]
                avg_hidden = verl_F.masked_mean(prompts_last_hidden, prompt_attention_mask.unsqueeze(-1), axis=1)

                # avg_hidden = avg_hidden.detach()  # use detach() to stop gradient of proj_z to policy
                proj_z = self.actor_module.proj_z
                # Defensive alignment: proj_z may be excluded from FSDP sharding and end up
                # on a different device than the actor hidden states.
                proj_z_param = next(proj_z.parameters(), None)
                if proj_z_param is not None:
                    if proj_z_param.device != avg_hidden.device:
                        proj_z.to(device=avg_hidden.device)
                        proj_z_param = next(proj_z.parameters(), None)
                    if proj_z_param is not None and avg_hidden.dtype != proj_z_param.dtype:
                        avg_hidden = avg_hidden.to(dtype=proj_z_param.dtype)
                log_z = proj_z(avg_hidden)

                return entropy, log_probs, log_z
                
            else:
                return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        available_batch_keys = set(data.batch.keys())
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if "ref_log_prob" in available_batch_keys:
            select_keys.append("ref_log_prob")
        elif self.config.use_kl_loss:
            raise RuntimeError("`ref_log_prob` is required but missing from actor update batch.")
        if "rule_reward" in available_batch_keys:
            select_keys.append("rule_reward")
        if "token_level_rewards" in available_batch_keys:
            select_keys.append("token_level_rewards")
        for chi_key in (
            "chi_pos_input_ids",
            "chi_pos_attention_mask",
            "chi_pos_position_ids",
            "chi_pos_responses",
            "chi_pos_response_mask",
            "chi_pos_ref_log_prob",
        ):
            if chi_key in available_batch_keys:
                select_keys.append(chi_key)
        if multi_turn:
            select_keys.append("loss_mask")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    ref_log_prob = data["ref_log_prob"] if "ref_log_prob" in data else old_log_prob

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, log_z = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, return_log_z=True)

                    # Build current reward from current policy and detach log(pi) inside reward used by flow loss.
                    flow_reward, _, _ = self._compute_current_flow_reward(
                        logpf=log_prob,
                        logf_ref=ref_log_prob,
                        response_mask=response_mask,
                        sample_data=data,
                    )

                    flow_loss, flow_metrics = self.compute_flowrl_objective(
                        logpf=log_prob,
                        logf_ref=ref_log_prob,
                        logpf_old=old_log_prob,
                        log_z=log_z,
                        reward=flow_reward,
                        response_mask=response_mask,
                        clip_ratio=self.config.clip_ratio,
                    )

                    chi_squared_loss_coef = float(self.config.get("chi_squared_loss_coef", 1.0))
                    chi_required = bool(self.config.get("chi_squared_require_positive_batch", True))
                    has_chi_positive_batch = all(
                        key in data
                        for key in (
                            "chi_pos_input_ids",
                            "chi_pos_attention_mask",
                            "chi_pos_position_ids",
                            "chi_pos_responses",
                            "chi_pos_response_mask",
                            "chi_pos_ref_log_prob",
                        )
                    )
                    if chi_squared_loss_coef != 0.0 and not has_chi_positive_batch and chi_required:
                        raise RuntimeError(
                            "chi-squared loss is enabled but chi positive batch is missing. "
                            "Expected chi_pos_* tensors from trainer."
                        )

                    if chi_squared_loss_coef != 0.0 and has_chi_positive_batch:
                        pos_micro_batch = {
                            "input_ids": data["chi_pos_input_ids"],
                            "attention_mask": data["chi_pos_attention_mask"],
                            "position_ids": data["chi_pos_position_ids"],
                            "responses": data["chi_pos_responses"],
                        }
                        _, pos_log_prob = self._forward_micro_batch(
                            micro_batch=pos_micro_batch,
                            temperature=temperature,
                            calculate_entropy=False,
                            return_log_z=False,
                        )
                        chi_squared_loss, chi_metrics = self._compute_chi_squared_loss(
                            pos_logpf=pos_log_prob,
                            pos_logf_ref=data["chi_pos_ref_log_prob"],
                            pos_response_mask=data["chi_pos_response_mask"],
                            neg_logpf=log_prob,
                            neg_logf_ref=ref_log_prob,
                            neg_response_mask=response_mask,
                        )
                    else:
                        chi_squared_loss = flow_loss.new_zeros(())
                        chi_metrics = {
                            "actor/chi_squared_loss": 0.0,
                            "actor/chi_squared_loss_pos": 0.0,
                            "actor/chi_squared_loss_neg": 0.0,
                            "actor/chi_squared_loss_reg": 0.0,
                            "actor/chi_lratio_pos": 0.0,
                            "actor/chi_lratio_neg": 0.0,
                        }

                    policy_loss = flow_loss + chi_squared_loss_coef * chi_squared_loss

                    flow_metrics.update(chi_metrics)
                    flow_metrics.update(
                        {
                            "actor/chi_squared_loss_coef": chi_squared_loss_coef,
                            "actor/final_loss": policy_loss.detach().item(),
                        }
                    )

                    # pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    #     old_log_prob=old_log_prob,
                    #     log_prob=log_prob,
                    #     advantages=advantages,
                    #     response_mask=response_mask,
                    #     cliprange=clip_ratio,
                    #     cliprange_low=clip_ratio_low,
                    #     cliprange_high=clip_ratio_high,
                    #     clip_ratio_c=clip_ratio_c,
                    #     loss_agg_mode=loss_agg_mode,
                    # )

                    # if entropy_coeff != 0:
                        # entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    # compute policy loss
                    #     policy_loss = pg_loss - entropy_loss * entropy_coeff
                    # else:
                    #     policy_loss = pg_loss

                    # if self.config.use_kl_loss:
                    #     ref_log_prob = data["ref_log_prob"]
                    #     # compute kl loss
                    #     kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    #     kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    #     policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    #     metrics["actor/kl_loss"] = kl_loss.detach().item()
                    #     metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    # data['actor/entropy'] = entropy_loss.detach().item(),
                    # data = {
                    #     "actor/pg_loss": pg_loss.detach().item(),
                    #     "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    #     "actor/ppo_kl": ppo_kl.detach().item(),
                    #     "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    # }
                    append_to_dict(metrics, flow_metrics)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    def _compute_current_flow_reward(
        self,
        logpf: torch.Tensor,
        logf_ref: torch.Tensor,
        response_mask: torch.Tensor,
        sample_data: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chi_squared_beta = float(self.config.get("chi_squared_beta", 0.001))
        flow_reward_clip_min = self.config.get("flow_reward_clip_min", None)

        current_log_ratio = verl_F.masked_mean(logpf - logf_ref, response_mask, axis=1)

        if "rule_reward" in sample_data:
            rule_reward = sample_data["rule_reward"]
        elif "token_level_rewards" in sample_data:
            token_level_rewards = sample_data["token_level_rewards"]
            if token_level_rewards.ndim == 1:
                rule_reward = token_level_rewards
            else:
                rule_reward = verl_F.masked_sum(token_level_rewards, response_mask, axis=1)
        else:
            # Fall back to provided advantages if explicit rule reward is unavailable.
            if sample_data["advantages"].ndim == 1:
                rule_reward = sample_data["advantages"]
            else:
                rule_reward = verl_F.masked_mean(sample_data["advantages"], response_mask, axis=1)

        rule_reward = rule_reward.to(device=current_log_ratio.device, dtype=current_log_ratio.dtype)
        flow_reward = rule_reward + chi_squared_beta * current_log_ratio.detach()

        if flow_reward_clip_min is not None:
            flow_reward = torch.clamp_min(flow_reward, float(flow_reward_clip_min))

        return flow_reward, current_log_ratio, rule_reward

    def _compute_chi_squared_loss(
        self,
        pos_logpf: torch.Tensor,
        pos_logf_ref: torch.Tensor,
        pos_response_mask: torch.Tensor,
        neg_logpf: torch.Tensor,
        neg_logf_ref: torch.Tensor,
        neg_response_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # Match chi_squared_rm.py:
        #   0.5 * (beta * log(pi/pi_ref)(y+) - r_max)^2
        # + 0.5 * (beta * log(pi/pi_ref)(y-) - r_min)^2
        # + reg_coef * ((beta * log(pi/pi_ref)(y+))^2 + (beta * log(pi/pi_ref)(y-))^2)
        chi_squared_beta = float(self.config.get("chi_squared_beta", 0.001))
        chi_squared_reg_coef = float(self.config.get("chi_squared_reg_coef", 0.005))
        chi_squared_r_max = float(self.config.get("chi_squared_r_max", 1.0))
        chi_squared_r_min = float(self.config.get("chi_squared_r_min", -1.0))

        pos_response_mask = pos_response_mask.to(device=pos_logpf.device, dtype=pos_logpf.dtype)
        neg_response_mask = neg_response_mask.to(device=neg_logpf.device, dtype=neg_logpf.dtype)
        pos_logf_ref = pos_logf_ref.to(device=pos_logpf.device, dtype=pos_logpf.dtype)
        neg_logf_ref = neg_logf_ref.to(device=neg_logpf.device, dtype=neg_logpf.dtype)

        logp_pi_pos = verl_F.masked_sum(pos_logpf, pos_response_mask, axis=1)
        logp_ref_pos = verl_F.masked_sum(pos_logf_ref, pos_response_mask, axis=1)
        logp_pi_neg = verl_F.masked_sum(neg_logpf, neg_response_mask, axis=1)
        logp_ref_neg = verl_F.masked_sum(neg_logf_ref, neg_response_mask, axis=1)

        lr_pos = logp_pi_pos - logp_ref_pos
        lr_neg = logp_pi_neg - logp_ref_neg

        pred_pos = chi_squared_beta * lr_pos
        pred_neg = chi_squared_beta * lr_neg

        loss_pos = 0.5 * (pred_pos - chi_squared_r_max).pow(2)
        loss_neg = 0.5 * (pred_neg - chi_squared_r_min).pow(2)
        loss_reg = chi_squared_reg_coef * (pred_pos.pow(2) + pred_neg.pow(2))
        chi_squared_loss = (loss_pos + loss_neg + loss_reg).mean()

        chi_metrics = {
            "actor/chi_squared_loss": chi_squared_loss.detach().item(),
            "actor/chi_squared_loss_pos": loss_pos.mean().detach().item(),
            "actor/chi_squared_loss_neg": loss_neg.mean().detach().item(),
            "actor/chi_squared_loss_reg": loss_reg.mean().detach().item(),
            "actor/chi_squared_beta": chi_squared_beta,
            "actor/chi_squared_reg_coef": chi_squared_reg_coef,
            "actor/chi_squared_r_max": chi_squared_r_max,
            "actor/chi_squared_r_min": chi_squared_r_min,
            "actor/chi_lratio_pos": lr_pos.detach().mean().item(),
            "actor/chi_lratio_neg": lr_neg.detach().mean().item(),
        }
        return chi_squared_loss, chi_metrics

    def compute_flowrl_objective(self, logpf=None, logf_ref=None,  logpf_old=None, log_z=None, reward=None, response_mask=None, clip_ratio=None):

        log_z = log_z.squeeze(-1)

        # mean of log p_f / log p_ref over valid tokens
        avg_logpf = verl_F.masked_mean(logpf, response_mask, axis=1)
        avg_logp_ref = verl_F.masked_mean(logf_ref, response_mask, axis=1)

        if reward.ndim == 1:
            seq_log_reward = reward
        elif reward.ndim == 2:
            seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)
        else:
            raise ValueError(f"`reward` should be 1D or 2D tensor, got shape={tuple(reward.shape)}")
        seq_log_reward = seq_log_reward.to(device=avg_logpf.device, dtype=avg_logpf.dtype)

        if bool(self.config.get("detach_reward_for_flow_loss", True)):
            seq_log_reward = seq_log_reward.detach()
        
        # TB loss residual
        delta = log_z + avg_logpf - 15 * seq_log_reward - avg_logp_ref

        # important sampling
        log_w = verl_F.masked_sum(logpf - logpf_old, response_mask, axis=1)  # sum over valid tokens per trajectory
        imp_w_raw = torch.exp(log_w).detach() 
        imp_w = torch.clamp(imp_w_raw, max=10)

        weighted_losses = imp_w * (delta ** 2)
        avg_loss = torch.mean(weighted_losses)
        
        # Loss statistics and PPO-style metrics
        # Compute approximate KL divergence between current policy and reference policy
        approx_kl_ref = logpf - logf_ref  # KL(pi_f || pi_ref)
        negative_approx_kl = logpf - logpf_old  # KL(pi_f || pi_old) for policy change tracking

        # Policy ratio for reference policy (for monitoring distribution shift)
        # ratio_ref = torch.exp(approx_kl_ref)

        # Compute PPO KL and Reference KL for monitoring
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
        ref_kl = verl_F.masked_mean(-approx_kl_ref, response_mask)

        loss_term_dict = {
                            "actor/log_prob": verl_F.masked_mean(logpf, response_mask).detach().item(),
                            "actor/old_log_prob": verl_F.masked_mean(logpf_old, response_mask).detach().item(),
                            "actor/ref_log_prob": verl_F.masked_mean(logf_ref, response_mask).detach().item(),
                            "actor/log_z": log_z.mean().detach().item(),
                            "actor/log_reward": seq_log_reward.mean().detach().item(),
                            "actor/flow_loss": avg_loss.detach().item(),
                            "actor/importance_weight": imp_w.mean().detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),  # PPO-style KL (current vs old policy)
                            "actor/ref_kl": ref_kl.detach().item(),  # KL with reference policy
                        }
                        
        return avg_loss, loss_term_dict
