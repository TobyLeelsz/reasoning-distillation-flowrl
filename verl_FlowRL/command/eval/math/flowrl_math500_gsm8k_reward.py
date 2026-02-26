"""Reward function used by command/eval/math/flowrl_eval_math500_gsm8k.sh."""

from verl.utils.reward_score import gsm8k, math_dapo


def _math500_accuracy(solution_str: str, ground_truth: str) -> float:
    result = math_dapo.compute_score(solution_str=solution_str, ground_truth=ground_truth)
    if isinstance(result, dict):
        return float(bool(result.get("acc", False)))
    return float(result > 0)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source in ("openai/gsm8k", "gsm8k"):
        return float(gsm8k.compute_score(solution_str=solution_str, ground_truth=ground_truth))

    if data_source in (
        "math500",
        "HuggingFaceH4/MATH-500",
        "math_dapo",
        "lighteval/MATH",
        "DigitalLearningGmbH/MATH-lighteval",
    ):
        return _math500_accuracy(solution_str=solution_str, ground_truth=ground_truth)

    raise NotImplementedError(f"Unsupported data_source for eval script: {data_source}")
