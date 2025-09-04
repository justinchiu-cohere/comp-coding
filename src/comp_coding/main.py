from typing import List, Optional, Dict, Tuple, Any, Union
from pydantic import BaseModel, HttpUrl
from tqdm import tqdm
import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from multiprocessing import Pool, cpu_count
import itertools


class TestCase(BaseModel):
    stdin: str
    stdout: str


class ScenarioConfig(BaseModel):
    prompt: str
    tests: List[TestCase]
    test_type: str = "stdinout"
    reward_type: str = "binary"
    execution_server_url: HttpUrl = "http://codeserver-service.default:80"


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
hf_datasets: Dict[str, Union[Dataset, DatasetDict]] = {
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
        n_workers = min(cpu_count(), 64)  # Cap at 64 for safety

    # Check if cache exists
    if problems_path.exists() and mapping_path.exists():
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

    # Collect items to process
    items_to_process = []
    for lang in ["python", "cpp"]:
        ocr2_ds = ocr2_dataset[lang]  # type: ignore

        if num_examples is not None:
            # Take first num_examples items across both languages
            remaining = max(0, num_examples - len(items_to_process))
            if remaining > 0:
                lang_items = list(itertools.islice(ocr2_ds, remaining))
                items_to_process.extend(lang_items)
        else:
            # Process all items
            items_to_process.extend(ocr2_ds)

    # Process items in parallel
    print(f"Processing {len(items_to_process)} OCR2 items in parallel...")
    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    process_ocr2_item_for_mapping, items_to_process, chunksize=100
                ),
                total=len(items_to_process),
                desc="Scanning for unique problems",
            )
        )

    # Aggregate results
    for result in results:
        if result is not None:
            question_id, problem_key = result
            question_id_to_problem_key[question_id] = problem_key

            if problem_key not in problem_key_to_question_ids:
                problem_key_to_question_ids[problem_key] = []
            if question_id not in problem_key_to_question_ids[problem_key]:
                problem_key_to_question_ids[problem_key].append(question_id)

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
        n_workers = min(cpu_count(), 64)  # Cap at 64 for safety

    print(f"\nAppending samples to problems using {n_workers} workers...")
    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

    # Collect items to process
    items_to_process = []
    for lang in ["python", "cpp"]:
        ocr2_ds = ocr2_dataset[lang]  # type: ignore

        if num_examples is not None:
            # Take first num_examples items across both languages
            remaining = max(0, num_examples - len(items_to_process))
            if remaining > 0:
                lang_items = list(itertools.islice(ocr2_ds, remaining))
                items_to_process.extend(lang_items)
        else:
            # Process all items
            items_to_process.extend(ocr2_ds)

    # Process items in parallel to create samples
    print(f"Processing {len(items_to_process)} OCR2 items to create samples...")
    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    process_ocr2_item_to_sample, items_to_process, chunksize=100
                ),
                total=len(items_to_process),
                desc="Creating samples",
            )
        )

    # Group samples by problem index (thread-safe aggregation)
    samples_by_index: Dict[int, List[Sample]] = {}
    samples_added = 0
    samples_skipped = 0

    for result in results:
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

    # Now safely append all samples to their problems
    for problem_index, samples in samples_by_index.items():
        problems[problem_index].samples.extend(samples)

    print(
        f"\nAdded {samples_added} samples to problems, skipped {samples_skipped} samples"
    )


def convert_ocr2_to_problems(
    problems_path: Path = Path("problems.json"),
    mapping_path: Path = Path("question_id_to_index.json"),
    num_examples: Optional[int] = None,
    n_workers: Optional[int] = None,
) -> List[Problem]:
    """
    Convert OpenCodeReasoning2 dataset items to Problem instances.

    Args:
        problems_path: Path to problems JSON file
        mapping_path: Path to question_id to index mapping file
        num_examples: Optional limit on number of examples to process (for testing)
        n_workers: Number of worker processes (defaults to CPU count)

    Returns:
        List of Problem instances with samples attached
    """

    # Step 1: Build or load the question_id to index mapping and problems
    question_id_to_index, problems = build_question_id_to_problem_mapping(
        problems_path, mapping_path, num_examples, n_workers
    )

    # Step 2: Append all samples to their corresponding problems
    append_samples_to_problems(question_id_to_index, problems, num_examples, n_workers)

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
        problems_path, mapping_path, args.num_examples, args.n_workers
    )

    print(f"\nConverted {len(problems)} unique problems")

    # Count total samples
    total_samples = sum(len(p.samples) for p in problems)
    print(f"Total samples across all problems: {total_samples}")

    if problems:
        print("\nFirst problem example:")
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
