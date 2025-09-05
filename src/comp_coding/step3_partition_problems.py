#!/usr/bin/env python3
"""
Step 3: Partition filtered problems and convert to training formats.
Creates 4 balanced partitions and generates aggregated SFT/RL training splits.
"""

from typing import List, Dict, Any
import json
from pathlib import Path
import argparse
import numpy as np
import random

# Import shared models
from comp_coding.models import (
    Problem,
    SFTSample,
    RLProblem,
    ScenarioConfig,
    TestCase,
)


def problem_to_sft_samples(problem: Problem) -> List[SFTSample]:
    """
    Convert a Problem to SFT samples.
    Uses the problem's prompt and each sample's solution.
    """
    sft_samples = []
    for sample in problem.samples:
        if sample.solution:
            sft_example = SFTSample(
                prompt=problem.scenario_config.prompt, solution=sample.solution
            )
            sft_samples.append(sft_example)
    return sft_samples


def problem_to_rl_format(problem: Problem) -> RLProblem:
    """
    Convert a Problem to RL format.
    """
    # Create ScenarioConfig with tests
    scenario_config = ScenarioConfig(
        prompt=problem.scenario_config.prompt, tests=problem.scenario_config.tests
    )

    # Create RLProblem
    rl_problem = RLProblem(env_name="code", scenario_config=scenario_config)

    return rl_problem


def partition_problems(
    problems: List[Problem], num_partitions: int = 4, random_seed: int = 42
) -> List[List[Problem]]:
    """
    Partition problems into equal-sized sets.
    Always shuffles problems before partitioning using numpy permutation.
    """
    print(f"\nPartitioning {len(problems)} problems into {num_partitions} equal-sized sets...")
    print(f"Using random seed: {random_seed}")

    # Shuffle problems using numpy permutation
    np.random.seed(random_seed)
    indices = np.random.permutation(len(problems))
    problems_shuffled = [problems[i] for i in indices]
    print(f"Shuffled {len(problems)} problems using numpy permutation")

    # Calculate partition size
    partition_size = len(problems) // num_partitions
    remainder = len(problems) % num_partitions

    partitions = []
    start = 0

    for i in range(num_partitions):
        # Add one extra problem to first 'remainder' partitions
        current_size = partition_size + (1 if i < remainder else 0)
        end = start + current_size
        partition = problems_shuffled[start:end]
        partitions.append(partition)
        print(f"  Partition {i + 1}: {len(partition)} problems (indices {start}-{end-1})")
        start = end

    return partitions


def create_aggregated_training_splits(
    partitions: List[List[Problem]],
    output_dir: Path,
    sft_ratios: List[float] = [0.75, 0.50, 0.25],
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Create aggregated SFT/RL training splits across partitions.
    For a 25-75 split with 4 partitions: partition 1 for SFT, partitions 2-4 for RL.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    num_partitions = len(partitions)
    all_stats = []
    
    # Create splits with different ratios
    for ratio in sft_ratios:
        split_name = f"{int(ratio * 100)}_{int((1 - ratio) * 100)}"
        print(f"\n  Creating aggregated {split_name} split...")
        
        # Calculate how many partitions for SFT vs RL
        num_sft_partitions = int(num_partitions * ratio)
        if num_sft_partitions == 0 and ratio > 0:
            num_sft_partitions = 1  # At least 1 partition for SFT if ratio > 0
        
        # Aggregate problems from appropriate partitions
        sft_problems = []
        rl_problems = []
        
        for i in range(num_partitions):
            if i < num_sft_partitions:
                sft_problems.extend(partitions[i])
            else:
                rl_problems.extend(partitions[i])
        
        # Convert to training formats
        sft_samples = []
        for problem in sft_problems:
            sft_samples.extend(problem_to_sft_samples(problem))
        
        rl_samples = []
        for problem in rl_problems:
            rl_samples.append(problem_to_rl_format(problem))
        
        # Save SFT samples
        sft_path = output_dir / f"split_{split_name}_sft.jsonl"
        with open(sft_path, "w") as f:
            for sample in sft_samples:
                f.write(json.dumps(sample.model_dump()) + "\n")
        
        # Save RL samples
        rl_path = output_dir / f"split_{split_name}_rl.jsonl"
        with open(rl_path, "w") as f:
            for sample in rl_samples:
                f.write(json.dumps(sample.model_dump()) + "\n")
        
        stats = {
            "split_name": split_name,
            "sft_partitions": list(range(1, num_sft_partitions + 1)),
            "rl_partitions": list(range(num_sft_partitions + 1, num_partitions + 1)),
            "sft_problems": len(sft_problems),
            "rl_problems": len(rl_problems),
            "sft_samples": len(sft_samples),
            "rl_samples": len(rl_samples),
            "sft_file": str(sft_path.name),
            "rl_file": str(rl_path.name),
        }
        all_stats.append(stats)
        
        print(f"    SFT: Partitions {stats['sft_partitions']} -> {len(sft_problems)} problems, {len(sft_samples)} samples")
        print(f"    RL:  Partitions {stats['rl_partitions']} -> {len(rl_problems)} problems, {len(rl_samples)} samples")
    
    # Create Luffy configuration (100% SFT AND 100% RL)
    print(f"\n  Creating Luffy split (100% SFT AND 100% RL)...")
    
    # Use ALL partitions for both SFT and RL
    all_problems = []
    for partition in partitions:
        all_problems.extend(partition)
    
    # Convert ALL problems to BOTH formats
    sft_samples = []
    rl_samples = []
    
    for problem in all_problems:
        sft_samples.extend(problem_to_sft_samples(problem))
        rl_samples.append(problem_to_rl_format(problem))
    
    # Save Luffy SFT samples
    sft_path = output_dir / f"split_luffy_sft.jsonl"
    with open(sft_path, "w") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample.model_dump()) + "\n")
    
    # Save Luffy RL samples
    rl_path = output_dir / f"split_luffy_rl.jsonl"
    with open(rl_path, "w") as f:
        for sample in rl_samples:
            f.write(json.dumps(sample.model_dump()) + "\n")
    
    stats = {
        "split_name": "luffy",
        "configuration": "100% SFT AND 100% RL (with clipped off-policy IS)",
        "sft_partitions": list(range(1, num_partitions + 1)),
        "rl_partitions": list(range(1, num_partitions + 1)),
        "sft_problems": len(all_problems),
        "rl_problems": len(all_problems),
        "sft_samples": len(sft_samples),
        "rl_samples": len(rl_samples),
        "sft_file": str(sft_path.name),
        "rl_file": str(rl_path.name),
    }
    all_stats.append(stats)
    
    print(f"    SFT: All partitions -> {len(all_problems)} problems, {len(sft_samples)} samples")
    print(f"    RL:  All partitions -> {len(all_problems)} problems (same problems for off-policy IS)")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Step 3: Partition and create training formats")
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input JSON file path (default: auto-detect based on num-examples)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix (default: auto-detect based on num-examples)",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=4,
        help="Number of partitions to create (default: 4)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for numpy permutation shuffling (default: 42)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples (used for filename detection)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Input directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation even if output files exist",
    )
    parser.add_argument(
        "--create-training-splits",
        action="store_true",
        help="Create aggregated SFT/RL training splits",
    )

    args = parser.parse_args()

    # Determine file paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.input_file:
        input_path = Path(args.input_file)
    else:
        prefix = (
            "problems" if not args.num_examples else f"problems_{args.num_examples}"
        )
        input_path = input_dir / f"{prefix}_step2_filtered.json"

    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        prefix = (
            "problems" if not args.num_examples else f"problems_{args.num_examples}"
        )
        output_prefix = f"{prefix}_step3_partition"

    # Check if any output files already exist
    existing_files = []
    for i in range(args.num_partitions):
        output_path = output_dir / f"{output_prefix}_{i + 1}.json"
        if output_path.exists():
            existing_files.append(output_path)

    if existing_files and not args.force_recreate:
        print(
            f"Output files already exist: {', '.join(str(f) for f in existing_files)}"
        )
        print("Use --force-recreate to overwrite.")
        return

    print("Step 3: Partitioning problems and creating training formats")
    print(f"Loading problems from {input_path}")

    # Load filtered problems
    with open(input_path, "r") as f:
        problems_data = json.load(f)
        problems = [Problem.model_validate(p) for p in problems_data]

    print(f"Loaded {len(problems)} problems")

    # Partition problems (always shuffles)
    partitions = partition_problems(
        problems, num_partitions=args.num_partitions, random_seed=args.random_seed
    )

    # Save each partition
    for i, partition in enumerate(partitions, 1):
        output_path = output_dir / f"{output_prefix}_{i}.json"
        print(f"\nSaving partition {i} to {output_path}")
        with open(output_path, "w") as f:
            partition_data = [prob.model_dump() for prob in partition]
            json.dump(partition_data, f, indent=2)

    # Create training splits if requested
    if args.create_training_splits:
        print("\n" + "=" * 60)
        print("Creating aggregated SFT/RL training splits...")
        print("=" * 60)

        # Create training directory
        training_dir = output_dir / "training_splits"
        training_dir.mkdir(exist_ok=True)

        # Create aggregated splits
        all_stats = create_aggregated_training_splits(
            partitions,
            training_dir,
            sft_ratios=[0.75, 0.50, 0.25],
            random_seed=args.random_seed,
        )

        # Save training statistics
        training_stats_path = training_dir / "training_splits_stats.json"
        with open(training_stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nTraining statistics saved to: {training_stats_path}")

    # Report partition statistics
    print("\n" + "=" * 60)
    print("Partition Statistics")
    print("=" * 60)

    for i, partition in enumerate(partitions, 1):
        total_samples = sum(len(p.samples) for p in partition)
        total_tests = sum(len(p.scenario_config.tests) for p in partition)

        # Dataset distribution in this partition
        dataset_counts = {}
        for p in partition:
            if p.dataset:
                dataset_counts[p.dataset] = dataset_counts.get(p.dataset, 0) + 1

        print(f"\nPartition {i}:")
        print(f"  Problems: {len(partition)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Total test cases: {total_tests}")
        print(
            f"  Datasets: {', '.join(f'{d}:{c}' for d, c in sorted(dataset_counts.items()))}"
        )

    # Save partition metadata
    metadata = {
        "num_partitions": args.num_partitions,
        "total_problems": len(problems),
        "random_seed": args.random_seed,
        "partitions": [],
    }

    for i, partition in enumerate(partitions, 1):
        partition_info = {
            "partition_id": i,
            "num_problems": len(partition),
            "num_samples": sum(len(p.samples) for p in partition),
            "num_tests": sum(len(p.scenario_config.tests) for p in partition),
            "datasets": {},
        }
        for p in partition:
            if p.dataset:
                partition_info["datasets"][p.dataset] = (
                    partition_info["datasets"].get(p.dataset, 0) + 1
                )
        metadata["partitions"].append(partition_info)

    metadata_path = output_dir / f"{output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nPartition metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()