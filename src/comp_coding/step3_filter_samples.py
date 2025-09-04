#!/usr/bin/env python3
"""
Step 3: Filter samples to keep only top 4 per problem.
Prioritizes by pass_rate (highest first), then by length (shortest first).
"""

from typing import List, Tuple
import json
from pathlib import Path
import argparse

# Import shared models
from models import Sample, Problem


def limit_samples_per_problem(problems: List[Problem], max_samples: int = 4) -> None:
    """
    Limit the number of samples per problem to max_samples.
    Prioritize by pass_rate (highest first), then by length (shortest first).

    Args:
        problems: List of Problem instances to limit samples for
        max_samples: Maximum number of samples per problem (default 4)
    """
    print(f"\nLimiting samples to {max_samples} per problem...")

    total_before = sum(len(p.samples) for p in problems)
    problems_limited = 0

    for problem in problems:
        if len(problem.samples) > max_samples:
            problems_limited += 1

            # Sort samples by pass_rate (descending) and then by length (ascending)
            # Calculate length as total characters in all string fields
            def sample_sort_key(sample: Sample) -> Tuple[float, int]:
                # Default pass_rate to 0 if None
                pass_rate = sample.pass_rate if sample.pass_rate is not None else 0.0
                # Calculate total length of all string fields
                length = 0
                if sample.r1_generation:
                    length += len(sample.r1_generation)
                if sample.qwq_critique:
                    length += len(sample.qwq_critique)
                if sample.solution:
                    length += len(sample.solution)
                if sample.judgement:
                    length += len(sample.judgement)
                # Return negative pass_rate for descending sort, positive length for ascending
                return (-pass_rate, length)

            # Sort and keep only top max_samples
            problem.samples.sort(key=sample_sort_key)
            problem.samples = problem.samples[:max_samples]

    total_after = sum(len(p.samples) for p in problems)

    print(
        f"Limited {problems_limited} problems from {total_before} to {total_after} total samples"
    )
    print(f"Removed {total_before - total_after} samples")


def report_statistics(problems: List[Problem]) -> None:
    """Report comprehensive statistics about the filtered problems."""
    # Comprehensive statistics
    total_samples = sum(len(p.samples) for p in problems)
    print(f"\nTotal samples across all problems: {total_samples}")

    # Dataset distribution
    print("\n=== Dataset Distribution ===")
    dataset_counts = {}
    dataset_sample_counts = {}
    for p in problems:
        if p.dataset:
            dataset_counts[p.dataset] = dataset_counts.get(p.dataset, 0) + 1
            dataset_sample_counts[p.dataset] = dataset_sample_counts.get(
                p.dataset, 0
            ) + len(p.samples)

    for dataset in sorted(dataset_counts.keys()):
        print(
            f"{dataset}: {dataset_counts[dataset]} problems, {dataset_sample_counts[dataset]} samples"
        )

    # Sample distribution statistics
    print("\n=== Sample Distribution ===")
    samples_per_problem = [len(p.samples) for p in problems]
    if samples_per_problem:
        print(f"Min samples per problem: {min(samples_per_problem)}")
        print(f"Max samples per problem: {max(samples_per_problem)}")
        print(
            f"Avg samples per problem: {sum(samples_per_problem) / len(samples_per_problem):.2f}"
        )

        # Problems with no samples
        no_sample_problems = sum(1 for s in samples_per_problem if s == 0)
        if no_sample_problems:
            print(f"Problems with no samples: {no_sample_problems}")

    # Test case statistics
    print("\n=== Test Case Statistics ===")
    test_counts = [len(p.scenario_config.tests) for p in problems]
    if test_counts:
        print(f"Min tests per problem: {min(test_counts)}")
        print(f"Max tests per problem: {max(test_counts)}")
        print(f"Avg tests per problem: {sum(test_counts) / len(test_counts):.2f}")

        no_test_problems = sum(1 for t in test_counts if t == 0)
        if no_test_problems:
            print(f"Problems with no tests: {no_test_problems}")

    # Sample quality statistics
    print("\n=== Sample Quality ===")
    r1_count = sum(1 for p in problems for s in p.samples if s.r1_generation)
    qwq_count = sum(1 for p in problems for s in p.samples if s.qwq_critique)
    solution_count = sum(1 for p in problems for s in p.samples if s.solution)
    judgement_count = sum(1 for p in problems for s in p.samples if s.judgement)

    if total_samples > 0:
        print(
            f"Samples with R1 generation: {r1_count} ({100 * r1_count / total_samples:.1f}%)"
        )
        print(
            f"Samples with QWQ critique: {qwq_count} ({100 * qwq_count / total_samples:.1f}%)"
        )
        print(
            f"Samples with solution: {solution_count} ({100 * solution_count / total_samples:.1f}%)"
        )
        print(
            f"Samples with judgement: {judgement_count} ({100 * judgement_count / total_samples:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Filter samples to keep only top 4 per problem"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=False,
        help="Input file with all samples (default: auto-detect based on num-examples)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="Output file for filtered problems (default: auto-detect based on num-examples)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Maximum samples per problem (default: 4)",
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
        help="Force recreation even if output file exists",
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
        input_path = input_dir / f"{prefix}_step2_all_samples.json"

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        prefix = (
            "problems" if not args.num_examples else f"problems_{args.num_examples}"
        )
        output_path = output_dir / f"{prefix}_step3_filtered.json"

    # Check if output already exists
    if output_path.exists() and not args.force_recreate:
        print(
            f"Output file {output_path} already exists. Use --force-recreate to overwrite."
        )
        return

    print(f"Step 3: Filtering samples to top {args.max_samples} per problem")
    print(f"Loading problems from {input_path}")

    # Load problems with all samples
    with open(input_path, "r") as f:
        problems_data = json.load(f)
        problems = [Problem.model_validate(p) for p in problems_data]

    print(f"Loaded {len(problems)} problems")

    # Filter samples
    limit_samples_per_problem(problems, max_samples=args.max_samples)

    # Save filtered problems
    print(f"\nSaving filtered problems to {output_path}")
    with open(output_path, "w") as f:
        problems_data = [prob.model_dump() for prob in problems]
        json.dump(problems_data, f, indent=2)

    # Report statistics
    report_statistics(problems)

    # Save statistics to file
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    total_samples = sum(len(p.samples) for p in problems)
    samples_per_problem = [len(p.samples) for p in problems]
    test_counts = [len(p.scenario_config.tests) for p in problems]

    stats = {
        "total_problems": len(problems),
        "total_samples": total_samples,
        "max_samples_per_problem": args.max_samples,
        "sample_distribution": {
            "min": min(samples_per_problem) if samples_per_problem else 0,
            "max": max(samples_per_problem) if samples_per_problem else 0,
            "avg": sum(samples_per_problem) / len(samples_per_problem)
            if samples_per_problem
            else 0,
        },
        "test_distribution": {
            "min": min(test_counts) if test_counts else 0,
            "max": max(test_counts) if test_counts else 0,
            "avg": sum(test_counts) / len(test_counts) if test_counts else 0,
        },
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
