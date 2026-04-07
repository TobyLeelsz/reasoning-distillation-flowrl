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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def _resolve_ray_temp_dir(config) -> str:
    # Keep Ray base temp dir short to avoid AF_UNIX socket path overflow (107 bytes).
    max_base_len = 27  # conservative: reserve ~80 chars for Ray session/socket suffix
    configured = config.ray_init.get("temp_dir", None) if "ray_init" in config else None
    candidates = [
        configured,
        os.environ.get("VERL_RAY_TMPDIR"),
        os.environ.get("RAY_TMPDIR"),
        f"/tmp/ray_{os.getuid()}",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(str(candidate)))
        if len(path) > max_base_len:
            print(f"[WARN] Skip Ray temp dir (too long for UNIX sockets): {path}")
            continue
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            print(f"[WARN] Skip Ray temp dir (cannot create): {path} ({exc})")
            continue
        return path

    fallback = "/tmp/ray"
    os.makedirs(fallback, exist_ok=True)
    return fallback


def run_ppo(config) -> None:
    if not ray.is_initialized():
        runtime_env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        }
        # Do not hardcode proxy settings. Inherit only when explicitly set in
        # the launcher environment.
        for key in [
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "no_proxy",
            "NO_PROXY",
            "CUDA_HOME",
            "CUDA_PATH",
            "DS_SKIP_CUDA_CHECK",
        ]:
            value = os.environ.get(key)
            if value:
                runtime_env_vars[key] = value

        ray_init_kwargs = {
            # address="auto",
            "_temp_dir": _resolve_ray_temp_dir(config),
            "runtime_env": {"env_vars": runtime_env_vars},
        }
        if config.ray_init.get("num_cpus", None) is not None:
            ray_init_kwargs["num_cpus"] = config.ray_init.num_cpus
        ray.init(**ray_init_kwargs)
        print(f"[INFO] Ray temp dir: {ray_init_kwargs['_temp_dir']}")

    print("[INFO] ✅ Ray has been initialized")
    print("[INFO] Available resources:")

    import pprint
    pprint.pprint(ray.available_resources())

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from copy import deepcopy
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_config = config
        if config.reward_model.get("use_rule_reward_for_eval", False):
            val_reward_config = deepcopy(config)
            with open_dict(val_reward_config):
                val_reward_config.reward_model.reward_manager = "naive"
                val_reward_config.custom_reward_function.path = None
            print("[INFO] Validation reward forced to rule-based (naive reward manager).")
        val_reward_fn = load_reward_manager(val_reward_config, tokenizer, num_examine=1, **val_reward_config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_data_config = deepcopy(config.data)
        val_data_config = deepcopy(config.data)
        with open_dict(val_data_config):
            # Keep rollout/training prompt formatting, but preserve original prompts for evaluation.
            val_data_config.system_prompt = None
            val_data_config.user_prompt_template = None

        train_dataset = create_rl_dataset(config.data.train_files, train_data_config, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, val_data_config, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
