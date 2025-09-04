#!/usr/bin/env python3
"""
Step 1: Create blank problems and mapping from OCR2 dataset.
This creates problems without samples, just the problem definitions and test cases.
"""

from typing import List, Optional, Dict, Tuple, Any
from tqdm import tqdm
import json
from pathlib import Path
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import itertools
import sys
import argparse

# Import shared types
from models import TestCase, ScenarioConfig, Problem

# Increase the limit for integer string conversion to handle large numbers in APPS dataset
sys.set_int_max_str_digits(100000)


# Load datasets globally for multiprocessing
print("Loading HuggingFace datasets...")
hf_datasets: Dict[str, Any] = {
    "taco": load_dataset("BAAI/TACO"),
    "apps": load_dataset("codeparrot/apps"),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces"),
}


def process_code_contests(
    benchmark: Dict[str, Any],
) -> Tuple[Optional[str], List[TestCase]]:
    """Process code_contests dataset item."""
    if not benchmark.get("description"):
        return None, []

    question = benchmark["description"]
    tests = []

    # Extract tests from public_tests and private_tests
    if benchmark.get("public_tests"):
        for inp, out in zip(
            benchmark["public_tests"]["input"], benchmark["public_tests"]["output"]
        ):
            tests.append(TestCase(stdin=inp, stdout=out))

    if benchmark.get("private_tests"):
        for inp, out in zip(
            benchmark["private_tests"]["input"],
            benchmark["private_tests"]["output"],
        ):
            tests.append(TestCase(stdin=inp, stdout=out))

    return question, tests


def process_taco(benchmark: Dict[str, Any]) -> Tuple[Optional[str], List[TestCase]]:
    """Process TACO dataset item."""
    question = benchmark.get("question", "")
    tests = []

    if benchmark.get("input_output"):
        io_data = json.loads(benchmark["input_output"])
        if "inputs" in io_data and "outputs" in io_data:
            for inp, out in zip(io_data["inputs"], io_data["outputs"]):
                # Check if inp and out are lists and join them
                if isinstance(inp, list):
                    inp = "\n".join(inp)
                if isinstance(out, list):
                    out = "\n".join(out)
                tests.append(TestCase(stdin=inp, stdout=out))

    return question, tests


def process_apps(benchmark: Dict[str, Any]) -> Tuple[Optional[str], List[TestCase]]:
    """Process APPS dataset item."""
    question = benchmark.get("question", "")
    tests = []

    # Handle APPS test cases - be careful with list inputs/outputs
    if benchmark.get("input_output"):
        io_data = json.loads(benchmark["input_output"])
        if "inputs" in io_data and "outputs" in io_data:
            for inp, out in zip(io_data["inputs"], io_data["outputs"]):
                # Check if inp and out are lists and join them
                if isinstance(inp, list):
                    inp = "\n".join(inp)
                if isinstance(out, list):
                    out = "\n".join(out)
                tests.append(TestCase(stdin=inp, stdout=out))

    return question, tests


def process_open_r1_codeforces(
    benchmark: Dict[str, Any],
) -> Tuple[Optional[str], List[TestCase]]:
    """Process open-r1/codeforces dataset item."""
    question = benchmark.get("problem", "")
    tests = []

    # Extract tests from tests_inputs and tests_outputs
    if benchmark.get("tests_inputs") and benchmark.get("tests_outputs"):
        for inp, out in zip(benchmark["tests_inputs"], benchmark["tests_outputs"]):
            tests.append(TestCase(stdin=inp, stdout=out))

    return question, tests


def get_question_and_tests(
    ds_name: str, ds_split: str, ds_index: int
) -> Tuple[Optional[str], List[TestCase]]:
    """Get question and tests from the original dataset."""
    if ds_name not in hf_datasets:
        raise ValueError(f"Unknown dataset: {ds_name}")

    dataset = hf_datasets[ds_name]
    if ds_split not in dataset:
        raise ValueError(f"Unknown split {ds_split} for dataset {ds_name}")

    if ds_index >= len(dataset[ds_split]):
        raise ValueError(
            f"Index {ds_index} out of range for {ds_name}[{ds_split}] which has {len(dataset[ds_split])} items"
        )

    benchmark = dataset[ds_split][ds_index]

    # Dispatch to the appropriate processor
    processors = {
        "code_contests": process_code_contests,
        "taco": process_taco,
        "apps": process_apps,
        "open-r1/codeforces": process_open_r1_codeforces,
    }

    processor = processors.get(ds_name)
    if processor:
        return processor(benchmark)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")


def process_ocr2_item_for_mapping(
    item: Dict[str, Any],
) -> Optional[Tuple[str, Tuple[str, str, int]]]:
    """Process a single OCR2 item to extract problem key and question_id.
    Returns (question_id, problem_key) or None."""

    ds_name: str = item["dataset"]
    ds_split: str = item["split"]
    ds_index: int = int(item["index"])
    question_id: str = item.get("question_id", "")

    if question_id:
        problem_key = (ds_name, ds_split, ds_index)
        return (question_id, problem_key)
    return None


def build_question_id_to_problem_mapping(
    problems_path: Path = Path("problems_blank.json"),
    mapping_path: Path = Path("question_id_to_index.json"),
    num_examples: Optional[int] = None,
    n_workers: Optional[int] = None,
    force_recreate: bool = False,
) -> Tuple[Dict[str, int], List[Problem]]:
    """
    Build mapping from question_id to problem index.
    Load from cache if it exists, otherwise build it and save to cache.

    Args:
        problems_path: Path to problems JSON file
        mapping_path: Path to question_id to index mapping file
        num_examples: Optional limit on number of examples to process (for testing)
        n_workers: Number of worker processes (defaults to CPU count)

    Returns:
        Tuple of (question_id_to_index dict, list of all problems)
    """

    if n_workers is None:
        n_workers = min(cpu_count(), 128)  # Use all available CPUs

    # Check if cache exists and not forcing recreation
    if not force_recreate and problems_path.exists() and mapping_path.exists():
        print(f"Loading cached data from {problems_path} and {mapping_path}")

        # Load problems
        with open(problems_path, "r") as f:
            problems_data = json.load(f)
            problems = [Problem(**prob_data) for prob_data in problems_data]

        # Load mapping
        with open(mapping_path, "r") as f:
            question_id_to_index = json.load(f)
            # Convert string keys back to int values
            question_id_to_index = {k: int(v) for k, v in question_id_to_index.items()}

        return question_id_to_index, problems

    print(f"Building question_id to Problem mapping using {n_workers} workers...")

    # First, collect all unique question_ids to identify unique problems
    question_id_to_problem_key: Dict[str, Tuple[str, str, int]] = {}
    problem_key_to_question_ids: Dict[Tuple[str, str, int], List[str]] = {}

    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

    # Process items using streaming with batches
    print("Processing OCR2 items in batches...")

    batch_size = 10000
    total_processed = 0

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

        print(f"Processing {lang} split (~{estimated_total} items)...")

        # Process in batches to avoid memory issues
        with Pool(n_workers) as pool:
            with tqdm(
                total=estimated_total, desc=f"Scanning {lang} for unique problems"
            ) as pbar:
                while True:
                    # Get next batch
                    batch = list(itertools.islice(item_iterator, batch_size))
                    if not batch:
                        break

                    # Process batch in parallel
                    batch_results = pool.map(process_ocr2_item_for_mapping, batch)  # type: ignore

                    # Aggregate batch results
                    for result in batch_results:
                        if result is not None:
                            question_id, problem_key = result
                            question_id_to_problem_key[question_id] = problem_key

                            if problem_key not in problem_key_to_question_ids:
                                problem_key_to_question_ids[problem_key] = []
                            if (
                                question_id
                                not in problem_key_to_question_ids[problem_key]
                            ):
                                problem_key_to_question_ids[problem_key].append(
                                    question_id
                                )

                    # Update progress
                    pbar.update(len(batch))
                    total_processed += len(batch)

                    if num_examples and total_processed >= num_examples:
                        break

    # Create Problem instances
    print(f"\nCreating {len(problem_key_to_question_ids)} unique problems...")
    problems: List[Problem] = []
    question_id_to_index: Dict[str, int] = {}

    for problem_key, question_ids in tqdm(
        problem_key_to_question_ids.items(), desc="Creating problems"
    ):
        ds_name, ds_split, ds_index = problem_key

        # Get question and tests from original dataset
        question, tests = get_question_and_tests(ds_name, ds_split, ds_index)

        if question is None:
            print(f"Skipping {ds_name}[{ds_split}][{ds_index}]: no question found")
            continue

        # Create ScenarioConfig
        scenario_config = ScenarioConfig(prompt=question, tests=tests if tests else [])

        # Create Problem with question_ids but no samples
        problem = Problem(
            scenario_config=scenario_config,
            dataset=ds_name,
            split=ds_split,
            index=ds_index,
            question_ids=question_ids,  # Store all question_ids for this problem
            samples=[],  # No samples yet
        )

        # Store the problem
        problem_index = len(problems)
        problems.append(problem)

        # Map all question_ids to this problem's index
        for qid in question_ids:
            question_id_to_index[qid] = problem_index

    # Save problems to separate file
    print(f"\nSaving {len(problems)} blank problems to {problems_path}")
    problems_path.parent.mkdir(parents=True, exist_ok=True)
    with open(problems_path, "w") as f:
        problems_data = [prob.model_dump() for prob in problems]
        json.dump(problems_data, f, indent=2)

    # Save mapping to separate file
    print(f"Saving question_id to index mapping to {mapping_path}")
    with open(mapping_path, "w") as f:
        json.dump(question_id_to_index, f, indent=2)

    return question_id_to_index, problems


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Create blank problems and mapping from OCR2 dataset"
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
        "--force-recreate",
        action="store_true",
        help="Force recreation even if cache exists",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for files",
    )

    args = parser.parse_args()

    # Set up output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    prefix = "problems" if not args.num_examples else f"problems_{args.num_examples}"
    problems_path = output_dir / f"{prefix}_step1_blank.json"
    mapping_path = output_dir / f"{prefix}_step1_mapping.json"

    print("Step 1: Creating blank problems and mapping")
    if args.num_examples:
        print(f"Processing limited to {args.num_examples} examples")
    else:
        print("Processing all examples")

    # Build the mapping and blank problems
    question_id_to_index, problems = build_question_id_to_problem_mapping(
        problems_path,
        mapping_path,
        args.num_examples,
        args.n_workers,
        args.force_recreate,
    )

    print(f"\nCreated {len(problems)} unique blank problems")
    print(f"Mapped {len(question_id_to_index)} question IDs")

    # Basic statistics
    dataset_counts = {}
    for p in problems:
        if p.dataset:
            dataset_counts[p.dataset] = dataset_counts.get(p.dataset, 0) + 1

    print("\nDataset distribution:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count} problems")


if __name__ == "__main__":
    main()
