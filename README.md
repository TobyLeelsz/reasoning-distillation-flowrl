# FlowRL: Matching Reward Distributions for LLM Reasoning

FlowRL is a flow-balanced reinforcement learning method that matches full reward distributions instead of maximizing rewards, promoting diverse exploration and generalizable reasoning trajectories in large language models.

![FlowRL Overview](figures/flowrl.png)

## Quick Start

### Installation

Install [veRL](https://github.com/volcengine/verl) first before using FlowRL.

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
cd verl@FlowRL

# For 7B math training
bash command/training/math/flowrl_7B_math.sh

# For 32B math training
bash command/training/math/flowrl_32B_math.sh

# For 7B code training
bash command/training/code/flowrl_7B_code.sh
```

### Testing

```bash
cd verl@Test

# First merge the model
bash command/eval/merge_model.sh

# For math testing
bash command/eval/math/flowrl_math_test.sh

# For code testing
bash command/eval/code/flowrl_code_test.sh
```
