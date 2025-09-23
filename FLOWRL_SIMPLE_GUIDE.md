# FlowRL Implementation

## 4 Simple Steps to Add FlowRL

### Step 1: Add Partition Function Z

**File**: `verl/workers/fsdp_workers.py`

[Add this class at line 100](https://github.com/Xuekai-Zhu/FlowRL/blob/5d4795bddd49d4a7f0d78a742b1c6bcd8bdec581/verl_FlowRL/verl/workers/fsdp_workers.py#L100):

```python
class ProjZModule(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Dropout(dropout)
            ])
        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

[Add this to model building at line 267](https://github.com/Xuekai-Zhu/FlowRL/blob/5d4795bddd49d4a7f0d78a742b1c6bcd8bdec581/verl_FlowRL/verl/workers/fsdp_workers.py#L265):

```python
n_dim = actor_module.config.hidden_size
actor_module.proj_z = ProjZModule(n_dim, num_layers=self.config.actor.proj_layer)
```

### Step 2: Modify Forward Pass

**File**: `verl/workers/actor/dp_actor.py`

[Change method signature at line 75](https://github.com/Xuekai-Zhu/FlowRL/blob/5d4795bddd49d4a7f0d78a742b1c6bcd8bdec581/verl_FlowRL/verl/workers/actor/dp_actor.py#L75):

```python
def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_log_z=False):
```

[Add before return at line 232](https://github.com/Xuekai-Zhu/FlowRL/blob/5d4795bddd49d4a7f0d78a742b1c6bcd8bdec581/verl_FlowRL/verl/workers/actor/dp_actor.py#L232):

```python
if return_log_z:
    last_hidden = output.hidden_states[-1].squeeze(0)
    # Handle padding/unpadding if using remove_padding...
    avg_hidden = verl_F.masked_mean(prompts_last_hidden, prompt_attention_mask.unsqueeze(-1), axis=1)
    log_z = self.actor_module.proj_z(avg_hidden)
    return entropy, log_probs, log_z
else:
    return entropy, log_probs
```

### Step 3: Replace PPO Loss with FlowRL Loss

**File**: `verl/workers/actor/dp_actor.py`

[Replace PPO loss computation around line 412](https://github.com/Xuekai-Zhu/FlowRL/blob/5d4795bddd49d4a7f0d78a742b1c6bcd8bdec581/verl_FlowRL/verl/workers/actor/dp_actor.py#L412):

```python
# OLD PPO CODE - REMOVE:
# entropy, log_prob = self._forward_micro_batch(...)
# pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(...)

# NEW FLOWRL CODE:
entropy, log_prob, log_z = self._forward_micro_batch(
    micro_batch=data, temperature=temperature,
    calculate_entropy=calculate_entropy, return_log_z=True
)

policy_loss, metrics_dict = self.compute_flowrl_objective(
    logpf=log_prob, logf_ref=data['ref_log_prob'], logpf_old=old_log_prob,
    log_z=log_z, reward=advantages, response_mask=response_mask,
    clip_ratio=self.config.clip_ratio
)
```

[Add FlowRL objective function at line 555](https://github.com/Xuekai-Zhu/FlowRL/blob/5d4795bddd49d4a7f0d78a742b1c6bcd8bdec581/verl_FlowRL/verl/workers/actor/dp_actor.py#L555):

```python
def compute_flowrl_objective(self, logpf, logf_ref, logpf_old, log_z, reward, response_mask, clip_ratio):
    log_z = log_z.squeeze(-1)
    avg_logpf = verl_F.masked_mean(logpf, response_mask, axis=1)
    avg_logp_ref = verl_F.masked_mean(logf_ref, response_mask, axis=1)
    seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

    # Trajectory balance: log_z + log_pf - 15*reward - log_p_ref
    delta = log_z + avg_logpf - 15 * seq_log_reward - avg_logp_ref

    # Importance sampling
    log_w = verl_F.masked_sum(logpf - logpf_old, response_mask, axis=1)
    importance_weight = torch.exp(log_w).detach()

    # Final loss
    weighted_losses = importance_weight * (delta ** 2)
    avg_loss = torch.mean(weighted_losses)

    metrics = {"actor/tb_loss": avg_loss.detach().item(),
               "actor/log_z": log_z.mean().detach().item()}
    return avg_loss, metrics
```

### Step 4: Fix Model Loading

**File**: `verl/workers/sharding_manager/fsdp_vllm.py`

[Change line 290-293](https://github.com/Xuekai-Zhu/FlowRL/blob/5d4795bddd49d4a7f0d78a742b1c6bcd8bdec581/verl_FlowRL/verl/workers/sharding_manager/fsdp_vllm.py#L290):

```python
# Skip proj_z parameters when loading to vLLM
loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                                            for name, param in updated_params.items()
                                            if not name.startswith("proj_z"))
                                            )
```