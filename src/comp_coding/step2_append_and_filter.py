#!/usr/bin/env python3
"""
Step 2: Append samples from OCR2 dataset and filter to top 4 per problem.
Combines appending and filtering to avoid storing all samples.
"""

from typing import List, Optional, Dict, Tuple, Any
from tqdm import tqdm
import json
from pathlib import Path
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import itertools
import argparse

# Import shared models
from comp_coding.models import Sample, Problem


def process_ocr2_item_to_sample(item: Dict[str, Any]) -> Optional[Tuple[str, Sample]]:
    """Process single OCR2 item to extract question_id and Sample."""
    if not item.get("question_id"):
        return None

    question_id = item["question_id"]

    # Create Sample from OCR2 item
    sample = Sample(
        r1_generation=item.get("r1_generation"),
        qwq_critique=item.get("qwq_critique"),
        solution=item.get("solution"),
        judgement=item.get("judgement"),
        pass_rate=item.get("pass_rate"),
        source=item.get("source"),
        license=item.get("license"),
        difficulty=item.get("difficulty"),
        id=item.get("id"),
        question_id=question_id,
    )

    return (question_id, sample)


def filter_samples(samples: List[Sample], max_samples: int = 4) -> List[Sample]:
    """
    Filter samples to keep only top max_samples.
    Prioritize by pass_rate (highest first), then by length (shortest first).
    """
    if len(samples) <= max_samples:
        return samples

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
    sorted_samples = sorted(samples, key=sample_sort_key)
    return sorted_samples[:max_samples]


def append_and_filter_samples(
    question_id_to_index: Dict[str, int],
    problems: List[Problem],
    num_examples: Optional[int] = None,
    n_workers: Optional[int] = None,
    max_samples: int = 4,
) -> None:
    """
    Append OCR2 samples to problems and filter to top max_samples per problem.

    Args:
        question_id_to_index: Mapping from question_id to problem index
        problems: List of Problem instances
        num_examples: Optional limit on number of examples to process
        n_workers: Number of worker processes
        max_samples: Maximum samples per problem (default 4)
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 128)

    print(f"\nAppending and filtering samples using {n_workers} workers...")
    print(f"Will keep top {max_samples} samples per problem")

    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

    # Process samples using streaming with batches
    print("Processing OCR2 samples in batches...")
    batch_size = 10000
    total_processed = 0
    samples_by_index: Dict[int, List[Sample]] = {}
    samples_added = 0
    samples_skipped = 0

    for lang in ["python"]:
        ocr2_ds = ocr2_dataset[lang]  # type: ignore

        # Create an iterator with limit if needed
        if num_examples is not None:
            remaining = num_examples - total_processed
            if remaining <= 0:
                break
            item_iterator = itertools.islice(ocr2_ds, remaining)
            estimated_total = min(remaining, len(ocr2_ds))
        else:
            item_iterator = iter(ocr2_ds)
            estimated_total = len(ocr2_ds)

        print(f"Processing {lang} samples (~{estimated_total} items)...")

        # Process in batches to avoid memory issues
        with Pool(n_workers) as pool:
            with tqdm(total=estimated_total, desc=f"Processing {lang} samples") as pbar:
                while True:
                    # Get next batch
                    batch = list(itertools.islice(item_iterator, batch_size))
                    if not batch:
                        break

                    # Process batch in parallel
                    batch_results = pool.map(process_ocr2_item_to_sample, batch)  # type: ignore

                    # Aggregate batch results
                    for result in batch_results:
                        if result is not None:
                            question_id, sample = result

                            if question_id in question_id_to_index:
                                problem_index = question_id_to_index[question_id]
                                if problem_index not in samples_by_index:
                                    samples_by_index[problem_index] = []
                                samples_by_index[problem_index].append(sample)
                                samples_added += 1
                            else:
                                samples_skipped += 1

                    # Update progress
                    pbar.update(len(batch))
                    total_processed += len(batch)

                    if num_examples and total_processed >= num_examples:
                        break

    # Filter and append samples to problems
    print(f"\nFiltering samples to top {max_samples} per problem...")
    total_before_filter = samples_added
    total_after_filter = 0
    problems_filtered = 0

    for problem_index, samples in samples_by_index.items():
        filtered_samples = filter_samples(samples, max_samples)
        problems[problem_index].samples.extend(filtered_samples)

        if len(samples) > max_samples:
            problems_filtered += 1
        total_after_filter += len(filtered_samples)

    print(f"Added {samples_added} samples total, skipped {samples_skipped}")
    print(f"Filtered {problems_filtered} problems, kept {total_after_filter} samples")
    print(f"Removed {total_before_filter - total_after_filter} samples")


def load_blank_problems_and_mapping(
    problems_path: Path,
    mapping_path: Path,
) -> Tuple[List[Problem], Dict[str, int]]:
    """Load blank problems and question_id to index mapping from step 1."""
    print(f"Loading blank problems from {problems_path}")
    with open(problems_path, "r") as f:
        problems_data = json.load(f)
        problems = [Problem.model_validate(p) for p in problems_data]

    print(f"Loading question_id to index mapping from {mapping_path}")
    with open(mapping_path, "r") as f:
        question_id_to_index = json.load(f)
        # Convert string indices back to int
        question_id_to_index = {k: int(v) for k, v in question_id_to_index.items()}

    print(f"Loaded {len(problems)} problems and {len(question_id_to_index)} mappings")
    return problems, question_id_to_index


def report_statistics(problems: List[Problem]) -> Dict:
    """Generate statistics about the problems and samples."""
    total_samples = sum(len(p.samples) for p in problems)

    # Dataset distribution
    dataset_counts = {}
    dataset_sample_counts = {}
    for p in problems:
        if p.dataset:
            dataset_counts[p.dataset] = dataset_counts.get(p.dataset, 0) + 1
            dataset_sample_counts[p.dataset] = dataset_sample_counts.get(
                p.dataset, 0
            ) + len(p.samples)

    # Sample distribution
    samples_per_problem = [len(p.samples) for p in problems]
    test_counts = [len(p.scenario_config.tests) for p in problems]

    # Sample quality
    r1_count = sum(1 for p in problems for s in p.samples if s.r1_generation)
    qwq_count = sum(1 for p in problems for s in p.samples if s.qwq_critique)
    solution_count = sum(1 for p in problems for s in p.samples if s.solution)
    judgement_count = sum(1 for p in problems for s in p.samples if s.judgement)

    stats = {
        "total_problems": len(problems),
        "total_samples": total_samples,
        "dataset_distribution": dataset_counts,
        "dataset_sample_distribution": dataset_sample_counts,
        "sample_distribution": {
            "min": min(samples_per_problem) if samples_per_problem else 0,
            "max": max(samples_per_problem) if samples_per_problem else 0,
            "avg": sum(samples_per_problem) / len(samples_per_problem)
            if samples_per_problem
            else 0,
            "no_samples": sum(1 for s in samples_per_problem if s == 0),
        },
        "test_distribution": {
            "min": min(test_counts) if test_counts else 0,
            "max": max(test_counts) if test_counts else 0,
            "avg": sum(test_counts) / len(test_counts) if test_counts else 0,
            "no_tests": sum(1 for t in test_counts if t == 0),
        },
        "sample_quality": {
            "r1_generation": r1_count,
            "qwq_critique": qwq_count,
            "solution": solution_count,
            "judgement": judgement_count,
        },
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Append and filter samples from OCR2 dataset"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to min(CPU count, 128))",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Maximum samples per problem (default: 4)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Input directory for blank problems",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for filtered problems",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation even if output file exists",
    )

    args = parser.parse_args()

    # Set up paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    prefix = "problems" if not args.num_examples else f"problems_{args.num_examples}"
    blank_problems_path = input_dir / f"{prefix}_step1_blank.json"
    mapping_path = input_dir / f"{prefix}_step1_mapping.json"
    output_path = output_dir / f"{prefix}_step2_filtered.json"
    stats_path = output_dir / f"{prefix}_step2_filtered_stats.json"

    # Check if output already exists
    if output_path.exists() and not args.force_recreate:
        print(
            f"Output file {output_path} already exists. Use --force-recreate to overwrite."
        )
        return

    print("Step 2: Appending and filtering samples")
    if args.num_examples:
        print(f"Processing limited to {args.num_examples} examples")
    else:
        print("Processing all examples")

    # Load blank problems and question_id to index mapping from step 1
    problems, question_id_to_index = load_blank_problems_and_mapping(
        blank_problems_path, mapping_path
    )

    # Append and filter samples in one pass
    append_and_filter_samples(
        question_id_to_index,
        problems,
        args.num_examples,
        args.n_workers,
        args.max_samples,
    )

    # Save filtered problems
    print(f"\nSaving filtered problems to {output_path}")
    with open(output_path, "w") as f:
        problems_data = [prob.model_dump() for prob in problems]
        json.dump(problems_data, f, indent=2)

    # Generate and save statistics
    stats = report_statistics(problems)

    # Print statistics
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Total problems: {stats['total_problems']}")

    print("\n=== Dataset Distribution ===")
    for dataset, count in sorted(stats["dataset_distribution"].items()):
        sample_count = stats["dataset_sample_distribution"][dataset]
        print(f"{dataset}: {count} problems, {sample_count} samples")

    print("\n=== Sample Distribution ===")
    sd = stats["sample_distribution"]
    print(f"Min samples per problem: {sd['min']}")
    print(f"Max samples per problem: {sd['max']}")
    print(f"Avg samples per problem: {sd['avg']:.2f}")
    if sd["no_samples"] > 0:
        print(f"Problems with no samples: {sd['no_samples']}")

    # Save statistics
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
