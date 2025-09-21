import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import openai
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
import os
import json

def calculate_diversity_metric(data_path, response_key="responses", success_threshold=0.5):
    """
    Calculate diversity metric from Equation 14.
    Diversity = (Σ S_i * I(S_i >= 1)) / (Σ I(S_i >= 1))
    """
    # Load data
    dataset = pd.read_parquet(data_path)
    responses = dataset["responses"]
    data_sources = dataset["data_source"]
    reward_model_data = dataset["reward_model"]
    
    # Extract unique successful trajectories per problem
    problem_trajectories = defaultdict(set)
    
    for i, response_list in enumerate(responses):
        for response in response_list:
            # Simple success check - modify based on your needs
            if len(response) > 10:  # Replace with actual success condition
                problem_trajectories[i].add(response.strip())
    
    # Calculate diversity
    numerator = sum(len(trajs) for trajs in problem_trajectories.values() if len(trajs) > 0)
    denominator = sum(1 for trajs in problem_trajectories.values() if len(trajs) > 0)
    
    diversity_score = numerator / denominator if denominator > 0 else 0.0
    
    # Results
    results = {
        "diversity_score": diversity_score,
        "total_problems": len(dataset),
        "problems_solved": len(problem_trajectories),
        "solve_rate": len(problem_trajectories) / len(dataset) if len(dataset) > 0 else 0,
        "total_unique_trajectories": numerator,
        "avg_trajectories_per_solved": diversity_score,
        "trajectory_distribution": {
            "min": min([len(t) for t in problem_trajectories.values()]) if problem_trajectories else 0,
            "max": max([len(t) for t in problem_trajectories.values()]) if problem_trajectories else 0,
            "mean": np.mean([len(t) for t in problem_trajectories.values()]) if problem_trajectories else 0
        }
    }
    
    # Save results
    output_path = data_path.replace('.parquet', '_diversity.json')
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Diversity Score: {diversity_score:.3f}")
    print(f"Problems Solved: {results['problems_solved']}/{results['total_problems']} ({results['solve_rate']:.1%})")
    print(f"Saved to: {output_path}")
    
    return results


def calculate_creativity_metric(data_path, response_key="responses"):
    """
    Calculate creativity metric for a single method.
    Measures the uniqueness of solutions within the dataset.
    """
    # Load data
    dataset = pd.read_parquet(data_path)
    responses = dataset[response_key]
    
    # Count solution frequency across all problems
    solution_counts = defaultdict(int)
    problem_solutions = defaultdict(list)
    
    for i, response_list in enumerate(responses):
        for response in response_list:
            if len(response) > 10:  # Replace with actual success condition
                solution = response.strip()
                solution_counts[solution] += 1
                problem_solutions[i].append(solution)
    
    # Calculate uniqueness metrics
    total_solutions = sum(solution_counts.values())
    unique_solutions = sum(1 for count in solution_counts.values() if count == 1)
    
    creativity_score = unique_solutions / total_solutions if total_solutions > 0 else 0.0
    
    # Results
    results = {
        "creativity_score": creativity_score,
        "total_solutions": total_solutions,
        "unique_solutions": unique_solutions,
        "duplicate_solutions": total_solutions - unique_solutions,
        "interpretation": f"{creativity_score:.1%} of solutions appear only once",
        "solution_frequency": {
            "appearing_once": unique_solutions,
            "appearing_twice": sum(1 for count in solution_counts.values() if count == 2),
            "appearing_3+_times": sum(1 for count in solution_counts.values() if count >= 3)
        }
    }
    
    # Save results
    output_path = data_path.replace('.parquet', '_creativity.json')
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Creativity Score: {creativity_score:.3f}")
    print(f"Unique Solutions: {unique_solutions}/{total_solutions} ({creativity_score:.1%})")
    print(f"Saved to: {output_path}")
    
    return results


def gpt_diversity_evaluation(data_path, response_key="responses", data_source_key="data_sources", 
                           sample_size=10, model="gpt-4o", api_key=None):
    """
    Use GPT to evaluate diversity of solution approaches.
    """
    if api_key:
        openai.api_key = api_key
    
    # Load data
    dataset = pd.read_parquet(data_path)
    responses = dataset[response_key]
    data_sources = dataset[data_source_key] if data_source_key in dataset.columns else [f"Problem {i}" for i in range(len(dataset))]
    
    # Filter problems with multiple responses
    valid_problems = [(i, data_sources[i], list(responses[i])) 
                     for i in range(len(dataset)) 
                     if len(responses[i]) > 1]
    
    # Sample problems
    if len(valid_problems) > sample_size:
        import random
        valid_problems = random.sample(valid_problems, sample_size)
    
    gpt_scores = []
    detailed_results = []
    
    for prob_idx, problem, solutions in tqdm(valid_problems, desc="GPT Evaluation"):
        prompt = f"""Analyze these solutions and count how many DISTINCT solution approaches are used.
                    Consider solutions the same if they use the same method, even with different details.

                    Problem: {problem}

                    Solutions:
                    {chr(10).join([f"{i+1}. {sol[:200]}..." if len(sol) > 200 else f"{i+1}. {sol}" for i, sol in enumerate(solutions[:5])])}

                    Return ONLY a number."""
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            distinct_count = int(response.choices[0].message.content.strip())
        except:
            distinct_count = len(set(solutions))  # Fallback
        
        gpt_scores.append(distinct_count)
        detailed_results.append({
            "problem_idx": prob_idx,
            "problem": problem[:100] + "..." if len(problem) > 100 else problem,
            "num_solutions": len(solutions),
            "distinct_approaches": distinct_count,
            "diversity_ratio": distinct_count / len(solutions) if len(solutions) > 0 else 0
        })
    
    # Results
    results = {
        "avg_distinct_approaches": np.mean(gpt_scores) if gpt_scores else 0,
        "total_evaluated": len(gpt_scores),
        "diversity_distribution": {
            "min": min(gpt_scores) if gpt_scores else 0,
            "max": max(gpt_scores) if gpt_scores else 0,
            "mean": np.mean(gpt_scores) if gpt_scores else 0,
            "std": np.std(gpt_scores) if gpt_scores else 0
        },
        "detailed_results": detailed_results
    }
    
    # Save results
    output_path = data_path.replace('.parquet', '_gpt_diversity.json')
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"GPT Diversity: {results['avg_distinct_approaches']:.2f} distinct approaches on average")
    print(f"Evaluated: {results['total_evaluated']} problems")
    print(f"Saved to: {output_path}")
    
    return results


def analyze_file(data_path, use_gpt=False, gpt_api_key=None):
    """
    Run all diversity analyses on a single file.
    
    Args:
        data_path: Path to the parquet file
        use_gpt: Whether to use GPT for diversity evaluation
        gpt_api_key: OpenAI API key
    """
    print("="*60)
    print(f"ANALYZING: {data_path}")
    print("="*60)
    
    # 1. Calculate diversity metric
    print("\n1. Diversity Metric:")
    diversity_results = calculate_diversity_metric(data_path)
    
    # 2. Calculate creativity metric
    print("\n2. Creativity Metric:")
    creativity_results = calculate_creativity_metric(data_path)
    
    # 3. GPT diversity evaluation (optional)
    if use_gpt:
        print("\n3. GPT Diversity Evaluation:")
        gpt_results = gpt_diversity_evaluation(data_path, api_key=gpt_api_key)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    
    return {
        "diversity": diversity_results,
        "creativity": creativity_results,
        "gpt": gpt_results if use_gpt else None
    }


if __name__ == "__main__":
    # Example usage for single file
    data_path = "/fs-computility/plm/shared/zhuxuekai/reasoning_flow/outputs/qwen_32b/GRPO_global_step_200/test-output-16.parquet"
    
    # Analyze single file
    results = analyze_file(
        data_path=data_path,
        # use_gpt=True,  # Set to False to skip GPT evaluation
        use_gpt=False,  # Set to False to skip GPT evaluation
        gpt_api_key="sk-cySsnCdGiI8jysaHtQjbuOkoCEch2JDxYbrLLbyJX7sJm8Mu"  # Replace with actual key
    )
    
    # Or analyze multiple files separately
    # files = ["sft_results.parquet", "ppo_results.parquet", "for_results.parquet"]
    # for file_path in files:
    #     print(f"\n\nProcessing {file_path}...")
    #     analyze_file(file_path, use_gpt=False)