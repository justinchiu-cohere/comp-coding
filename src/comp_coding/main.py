from typing import List, Optional, Dict, Tuple, Any
from pydantic import BaseModel
from tqdm import tqdm
import json
from pathlib import Path
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import itertools
import sys

# Increase the limit for integer string conversion to handle large numbers in APPS dataset
sys.set_int_max_str_digits(100000)


class TestCase(BaseModel):
    stdin: str
    stdout: str


class ScenarioConfig(BaseModel):
    prompt: str
    tests: List[TestCase]
    test_type: str = "stdinout"
    reward_type: str = "binary"
    execution_server_url: str = "http://codeserver-service.default:80"


class Sample(BaseModel):
    """Individual sample from OCR2 dataset."""

    r1_generation: Optional[str] = None
    qwq_critique: Optional[str] = None
    solution: Optional[str] = None
    judgement: Optional[str] = None
    pass_rate: Optional[float] = None
    source: Optional[str] = None
    license: Optional[str] = None
    difficulty: Optional[str] = None
    id: Optional[str] = None
    question_id: Optional[str] = None


class Problem(BaseModel):
    """Problem with test cases and multiple samples."""

    env_name: str = "code"
    scenario_config: ScenarioConfig
    # Problem metadata
    dataset: Optional[str] = None
    split: Optional[str] = None
    index: Optional[int] = None
    # All question_ids that map to this problem
    question_ids: List[str] = []
    # List of samples for this problem
    samples: List[Sample] = []


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
        try:
            io_data = json.loads(benchmark["input_output"])
            if io_data.get("inputs") and io_data.get("outputs"):
                for inp, out in zip(io_data["inputs"], io_data["outputs"]):
                    # Handle cases where inp/out might be lists
                    if isinstance(inp, list):
                        inp = "\n".join(str(x) for x in inp)
                    if isinstance(out, list):
                        out = "\n".join(str(x) for x in out)
                    tests.append(TestCase(stdin=str(inp), stdout=str(out)))
        except (json.JSONDecodeError, TypeError):
            pass

    return question, tests


def process_apps(benchmark: Dict[str, Any]) -> Tuple[Optional[str], List[TestCase]]:
    """Process APPS dataset item."""
    question = benchmark.get("question", "")
    tests = []

    if benchmark.get("input_output"):
        try:
            io_data = json.loads(benchmark["input_output"])
            if io_data.get("inputs") and io_data.get("outputs"):
                for inp, out in zip(io_data["inputs"], io_data["outputs"]):
                    # Handle cases where inp/out might be lists
                    if isinstance(inp, list):
                        inp = "\n".join(str(x) for x in inp)
                    if isinstance(out, list):
                        out = "\n".join(str(x) for x in out)
                    tests.append(TestCase(stdin=str(inp), stdout=str(out)))
        except (json.JSONDecodeError, TypeError):
            pass

    return question, tests


def process_open_r1_codeforces(
    benchmark: Dict[str, Any],
) -> Tuple[Optional[str], List[TestCase]]:
    """Process open-r1/codeforces dataset item."""
    if not benchmark.get("description"):
        return None, []

    question = benchmark["description"]
    tests = []

    if benchmark.get("input_format"):
        question += "\n\nInput\n\n" + benchmark["input_format"]
    if benchmark.get("output_format"):
        question += "\n\nOutput\n\n" + benchmark["output_format"]

    if benchmark.get("examples") and question:
        question += "\n\nExamples"
        for example in benchmark["examples"]:
            if "input" in example:
                question += "\n\nInput\n\n" + example["input"]
            if "output" in example:
                question += "\n\nOutput\n\n" + example["output"]
            # Also collect as test cases
            if "input" in example and "output" in example:
                tests.append(TestCase(stdin=example["input"], stdout=example["output"]))

    if benchmark.get("note"):
        question += "\n\nNote\n\n" + benchmark["note"]

    return question, tests


def get_question_and_tests(
    ds_name: str, split: str, index: int
) -> Tuple[Optional[str], List[TestCase]]:
    """Extract question and test cases from the original dataset."""
    benchmark: Dict[str, Any] = hf_datasets[ds_name][split][int(index)]

    # Dispatch to dataset-specific processor
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


def process_ocr2_item_to_sample(item: Dict[str, Any]) -> Optional[Tuple[str, Sample]]:
    """Process a single OCR2 item to create a Sample.
    Returns (question_id, Sample) or None."""

    question_id = item.get("question_id", "")

    if question_id:
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
    return None


def build_question_id_to_problem_mapping(
    problems_path: Path = Path("problems.json"),
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

        # Create Problem with question_ids
        problem = Problem(
            scenario_config=scenario_config,
            dataset=ds_name,
            split=ds_split,
            index=ds_index,
            question_ids=question_ids,  # Store all question_ids for this problem
            samples=[],
        )

        # Store the problem
        problem_index = len(problems)
        problems.append(problem)

        # Map all question_ids to this problem's index
        for qid in question_ids:
            question_id_to_index[qid] = problem_index

    # Save problems to separate file
    print(f"\nSaving {len(problems)} problems to {problems_path}")
    problems_path.parent.mkdir(parents=True, exist_ok=True)
    with open(problems_path, "w") as f:
        problems_data = [prob.model_dump() for prob in problems]
        json.dump(problems_data, f, indent=2)

    # Save mapping to separate file
    print(f"Saving question_id to index mapping to {mapping_path}")
    with open(mapping_path, "w") as f:
        # Convert int values to strings for JSON serialization
        json.dump(question_id_to_index, f, indent=2)

    return question_id_to_index, problems


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


def report_statistics(problems: List[Problem], problems_path: Path) -> None:
    """Report comprehensive statistics about the converted problems."""
    # Comprehensive statistics
    total_samples = sum(len(p.samples) for p in problems)
    print(f"Total samples across all problems: {total_samples}")

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
    no_sample_problems = 0
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
    no_test_problems = 0
    if test_counts:
        print(f"Min tests per problem: {min(test_counts)}")
        print(f"Max tests per problem: {max(test_counts)}")
        print(f"Avg tests per problem: {sum(test_counts) / len(test_counts):.2f}")

        no_test_problems = sum(1 for t in test_counts if t == 0)
        if no_test_problems:
            print(f"Problems with no tests: {no_test_problems}")

    # Question ID coverage
    print("\n=== Question ID Coverage ===")
    total_question_ids = sum(len(p.question_ids) for p in problems)
    print(f"Total question IDs mapped: {total_question_ids}")
    avg_qids = 0.0
    if problems:
        avg_qids = total_question_ids / len(problems)
        print(f"Avg question IDs per problem: {avg_qids:.2f}")

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

    # Save statistics to file
    stats_path = problems_path.parent / f"{problems_path.stem}_stats.json"
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
            "no_samples": no_sample_problems,
        },
        "test_distribution": {
            "min": min(test_counts) if test_counts else 0,
            "max": max(test_counts) if test_counts else 0,
            "avg": sum(test_counts) / len(test_counts) if test_counts else 0,
            "no_tests": no_test_problems,
        },
        "question_id_coverage": {
            "total": total_question_ids,
            "avg_per_problem": avg_qids,
        },
        "sample_quality": {
            "r1_generation": r1_count,
            "qwq_critique": qwq_count,
            "solution": solution_count,
            "judgement": judgement_count,
        },
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")


def load_cached_problems(
    problems_path: Path,
    mapping_path: Path,
) -> Tuple[Optional[Dict[str, int]], Optional[List[Problem]]]:
    """
    Load cached problems and mapping if they exist.

    Args:
        problems_path: Path to problems JSON file
        mapping_path: Path to mapping JSON file

    Returns:
        Tuple of (question_id_to_index, problems) or (None, None) if cache doesn't exist
    """
    if not problems_path.exists() or not mapping_path.exists():
        return None, None

    print(f"Loading cached problems from {problems_path}")
    try:
        with open(problems_path, "r") as f:
            problems_data = json.load(f)
            problems = [Problem.model_validate(p) for p in problems_data]

        with open(mapping_path, "r") as f:
            question_id_to_index = json.load(f)

        print(f"Loaded {len(problems)} cached problems")
        return question_id_to_index, problems
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None, None


def convert_ocr2_to_problems(
    problems_path: Path = Path("problems.json"),
    mapping_path: Path = Path("question_id_to_index.json"),
    num_examples: Optional[int] = None,
    n_workers: Optional[int] = None,
    force_recreate: bool = False,
) -> List[Problem]:
    """
    Convert OpenCodeReasoning2 dataset items to Problem instances.

    Args:
        problems_path: Path to problems JSON file
        mapping_path: Path to question_id to index mapping file
        num_examples: Optional limit on number of examples to process (for testing)
        n_workers: Number of worker processes (defaults to CPU count)
        force_recreate: Force recreation even if cache exists

    Returns:
        List of Problem instances with samples attached
    """

    # Try to load from cache if not forcing recreation
    if not force_recreate:
        question_id_to_index, problems = load_cached_problems(
            problems_path, mapping_path
        )
        if question_id_to_index is not None and problems is not None:
            # Check if samples need to be populated
            total_samples = sum(len(p.samples) for p in problems)
            if total_samples == 0:
                print("Cached problems found but no samples. Appending samples...")
                append_samples_to_problems(
                    question_id_to_index, problems, num_examples, n_workers
                )
                # Limit samples to 4 per problem
                limit_samples_per_problem(problems, max_samples=4)
                # Save updated problems with samples
                print(f"Saving updated problems with samples to {problems_path}")
                with open(problems_path, "w") as f:
                    problems_data = [prob.model_dump() for prob in problems]
                    json.dump(problems_data, f, indent=2)
            else:
                print(f"Using cached problems with {total_samples} existing samples")
                # Apply sample limit even to cached problems
                limit_samples_per_problem(problems, max_samples=4)
            return problems

    # Step 1: Build or load the question_id to index mapping and problems
    question_id_to_index, problems = build_question_id_to_problem_mapping(
        problems_path, mapping_path, num_examples, n_workers, force_recreate
    )

    # Step 2: Append all samples to their corresponding problems
    append_samples_to_problems(question_id_to_index, problems, num_examples, n_workers)

    # Step 3: Limit samples to 4 per problem
    limit_samples_per_problem(problems, max_samples=4)

    # Save the problems with samples
    print(f"Saving problems with samples to {problems_path}")
    with open(problems_path, "w") as f:
        problems_data = [prob.model_dump() for prob in problems]
        json.dump(problems_data, f, indent=2)

    return problems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert OCR2 dataset to Problem instances"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)",
    )
    parser.add_argument(
        "--cache-prefix",
        type=str,
        default="",
        help="Prefix for cache file names",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to min(CPU count, 64))",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of problems even if cache exists",
    )

    args = parser.parse_args()

    # Construct file paths based on prefix and num_examples
    if args.cache_prefix:
        prefix = args.cache_prefix
    else:
        prefix = (
            "problems" if not args.num_examples else f"problems_{args.num_examples}"
        )

    # Save to data/ directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    problems_path = data_dir / f"{prefix}.json"
    mapping_path = data_dir / f"{prefix}_mapping.json"

    if args.num_examples:
        print(f"Processing limited to {args.num_examples} examples")
    else:
        print("Processing all examples")

    problems = convert_ocr2_to_problems(
        problems_path,
        mapping_path,
        args.num_examples,
        args.n_workers,
        args.force_recreate,
    )

    print(f"\nConverted {len(problems)} unique problems")

    # Report statistics
    report_statistics(problems, problems_path)

    if problems:
        print("\n=== First Problem Example ===")
        print(f"Dataset: {problems[0].dataset}")
        print(
            f"Question IDs: {problems[0].question_ids[:3]}..."
            if len(problems[0].question_ids) > 3
            else problems[0].question_ids
        )
        print(f"Prompt: {problems[0].scenario_config.prompt[:200]}...")
        print(f"Number of tests: {len(problems[0].scenario_config.tests)}")
        print(f"Number of samples: {len(problems[0].samples)}")
        if problems[0].samples:
            print(
                f"First sample has R1 generation: {bool(problems[0].samples[0].r1_generation)}"
            )
