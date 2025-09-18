# FlowRL: Matching Reward Distributions for LLM Reasoning

FlowRL is a flow-balanced reinforcement learning method that matches full reward distributions instead of maximizing rewards, promoting diverse exploration and generalizable reasoning trajectories in large language models.

![FlowRL Overview](figures/flowrl.png)

## Quick Start

### Installation

The implementation is based on [veRL](https://github.com/volcengine/verl). You must meet all VERL requirements first before using FlowRL.

### Data Preparation

```bash
# Option 1: Download our pre-processed datasets directly.
bash preprocess/down_load_dataset.sh
```

```bash
# Option 2: Process Data from Source. 
Process data from original sources. 
```
For detailed processing instructions, see [data/README.md](data/README.md).

### Model Preparation

For Math Tasks: `Qwen/Qwen2.5-7B` (default in script) ; `Qwen/Qwen2.5-32B`

For Code Tasks: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

```bash
# Download default model (Qwen2.5-7B for math)
bash preprocess/down_load_model.sh

# For other models, modify MODEL_NAME in the script before running
```

### Training
```bash
cd verl@FlowRL/command/training/math
bash flowrl_7B_math.sh
```

### Testing
```bash
cd verl@FlowRL
python -m pytest tests/
```
