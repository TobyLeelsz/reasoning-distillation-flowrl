# FlowRL: Matching Reward Distributions for LLM Reasoning

FlowRL is a flow-balanced reinforcement learning method that matches full reward distributions instead of maximizing rewards, promoting diverse exploration and generalizable reasoning trajectories in large language models.

![FlowRL Overview](figures/flowrl.png)

## Quick Start

### Installation

The implementation is based on [veRL](https://github.com/volcengine/verl). You must meet all VERL requirements first before using FlowRL.

### Data Preparation

Prepare your training data according to your specific task requirements. The data should be formatted and preprocessed appropriately for the FlowRL training pipeline.

### Model Preparation

Download the required pre-trained model using the provided script:

```bash
bash data_preprocess/down_load_model.sh
```

This will download the Qwen2.5-7B model to `pre_trained_model/Qwen/Qwen2.5-7B/`.

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
