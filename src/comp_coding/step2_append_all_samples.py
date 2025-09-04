#!/usr/bin/env python3
"""
Step 2: Append all samples from OCR2 dataset to the blank problems.
Loads blank problems and mapping from step 1, then adds all R1 samples.
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
from models import Sample, Problem


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


def append_samples_to_problems(
    question_id_to_index: Dict[str, int],
    problems: List[Problem],
    num_examples: Optional[int] = None,
    n_workers: Optional[int] = None,
) -> None:
    """
    Append all OCR2 samples to their corresponding problems using the index mapping.

    Args:
        question_id_to_index: Mapping from question_id to problem index
        problems: List of Problem instances
        num_examples: Optional limit on number of examples to process (for testing)
        n_workers: Number of worker processes (defaults to CPU count)
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 128)  # Use all available CPUs

    print(f"\nAppending samples to problems using {n_workers} workers...")
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
            with tqdm(total=estimated_total, desc=f"Creating {lang} samples") as pbar:
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

    # Now safely append all samples to their problems
    for problem_index, samples in samples_by_index.items():
        problems[problem_index].samples.extend(samples)

    print(
        f"\nAdded {samples_added} samples to problems, skipped {samples_skipped} samples"
    )


def load_blank_problems_and_mapping(
    problems_path: Path,
    mapping_path: Path,
) -> Tuple[List[Problem], Dict[str, int]]:
    """Load blank problems and mapping from step 1."""
    print(f"Loading blank problems from {problems_path}")
    with open(problems_path, "r") as f:
        problems_data = json.load(f)
        problems = [Problem.model_validate(p) for p in problems_data]

    print(f"Loading mapping from {mapping_path}")
    with open(mapping_path, "r") as f:
        question_id_to_index = json.load(f)

    print(f"Loaded {len(problems)} problems and {len(question_id_to_index)} mappings")
    return problems, question_id_to_index


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Append all samples from OCR2 dataset to blank problems"
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
        "--input-dir",
        type=str,
        default="data",
        help="Input directory for blank problems",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for problems with samples",
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
    output_path = output_dir / f"{prefix}_step2_all_samples.json"

    # Check if output already exists
    if output_path.exists() and not args.force_recreate:
        print(
            f"Output file {output_path} already exists. Use --force-recreate to overwrite."
        )
        return

    print("Step 2: Appending all samples to problems")
    if args.num_examples:
        print(f"Processing limited to {args.num_examples} examples")
    else:
        print("Processing all examples")

    # Load blank problems and mapping from step 1
    problems, question_id_to_index = load_blank_problems_and_mapping(
        blank_problems_path, mapping_path
    )

    # Append all samples
    append_samples_to_problems(
        question_id_to_index, problems, args.num_examples, args.n_workers
    )

    # Save problems with all samples
    print(f"\nSaving problems with all samples to {output_path}")
    with open(output_path, "w") as f:
        problems_data = [prob.model_dump() for prob in problems]
        json.dump(problems_data, f, indent=2)

    # Statistics
    total_samples = sum(len(p.samples) for p in problems)
    print(f"\nTotal samples across all problems: {total_samples}")

    # Sample distribution
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


if __name__ == "__main__":
    main()
