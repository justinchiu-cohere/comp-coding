#!/usr/bin/env python3
"""
Step 4: Partition filtered problems into 4 sets.
Creates 4 balanced partitions of problems for parallel processing or evaluation.
"""

from typing import List
import json
from pathlib import Path
import argparse
import numpy as np

# Import shared models
from models import Problem


def partition_problems(
    problems: List[Problem], num_partitions: int = 4, random_seed: int = 42
) -> List[List[Problem]]:
    """
    Partition problems into num_partitions equal-sized sets using numpy permutation.
    Always shuffles the problems before partitioning.

    Args:
        problems: List of Problem instances to partition
        num_partitions: Number of partitions to create (default 4)
        random_seed: Random seed for shuffling (default 42)

    Returns:
        List of partitions, each containing a subset of problems
    """
    total_problems = len(problems)
    print(
        f"\nPartitioning {total_problems} problems into {num_partitions} equal-sized sets..."
    )

    # Always set random seed for reproducibility
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    # Always shuffle using numpy permutation
    permutation = np.random.permutation(total_problems)
    problems_shuffled = [problems[i] for i in permutation]
    print(f"Shuffled {total_problems} problems using numpy permutation")

    # Calculate equal partition sizes
    # Drop the last few problems if total is not divisible by num_partitions
    problems_per_partition = total_problems // num_partitions
    total_used = problems_per_partition * num_partitions

    if total_used < total_problems:
        print(
            f"Note: Dropping last {total_problems - total_used} problems to ensure equal partitions"
        )
        problems_shuffled = problems_shuffled[:total_used]

    # Create equal-sized partitions
    partitions = []
    for i in range(num_partitions):
        start_idx = i * problems_per_partition
        end_idx = (i + 1) * problems_per_partition

        partition = problems_shuffled[start_idx:end_idx]
        partitions.append(partition)

        print(
            f"  Partition {i + 1}: {len(partition)} problems (indices {start_idx}-{end_idx - 1})"
        )

    return partitions


def report_partition_statistics(partitions: List[List[Problem]]) -> None:
    """Report statistics about the partitions."""
    print("\n=== Partition Statistics ===")

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


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Partition filtered problems into 4 sets"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=False,
        help="Input file with filtered problems (default: auto-detect based on num-examples)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=False,
        help="Output prefix for partition files (default: auto-detect based on num-examples)",
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
        input_path = input_dir / f"{prefix}_step3_filtered.json"

    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        prefix = (
            "problems" if not args.num_examples else f"problems_{args.num_examples}"
        )
        output_prefix = f"{prefix}_step4_partition"

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

    print(f"Step 4: Partitioning problems into {args.num_partitions} sets")
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

    # Report statistics
    report_partition_statistics(partitions)

    # Save partition metadata
    metadata_path = output_dir / f"{output_prefix}_metadata.json"
    metadata = {
        "num_partitions": args.num_partitions,
        "total_problems": len(problems),
        "total_problems_used": sum(len(p) for p in partitions),
        "shuffled": True,  # Always shuffled
        "random_seed": args.random_seed,
        "partition_sizes": [len(p) for p in partitions],
        "partition_files": [
            f"{output_prefix}_{i + 1}.json" for i in range(args.num_partitions)
        ],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nPartition metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
