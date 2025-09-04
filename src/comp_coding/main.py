from typing import List, Optional, Dict, Tuple, Any, Union
from pydantic import BaseModel, HttpUrl
from tqdm import tqdm
import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict


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
    # List of samples for this problem
    samples: List[Sample] = []


hf_datasets: Dict[str, Union[Dataset, DatasetDict]] = {
    "taco": load_dataset("BAAI/TACO"),
    "apps": load_dataset("codeparrot/apps"),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces"),
}


def get_question_and_tests(
    ds_name: str, split: str, index: int
) -> Tuple[Optional[str], List[TestCase]]:
    """Extract question and test cases from the original dataset."""
    benchmark: Dict[str, Any] = hf_datasets[ds_name][split][int(index)]
    question: Optional[str] = None
    tests: List[TestCase] = []

    if ds_name == "code_contests":
        if not benchmark.get("description"):
            return None, []
        question = benchmark["description"]
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

    elif ds_name == "taco":
        question = benchmark.get("question", "")
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

    elif ds_name == "apps":
        question = benchmark.get("question", "")
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

    elif ds_name == "open-r1/codeforces":
        if not benchmark.get("description"):
            return None, []
        question = benchmark["description"]
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
                    tests.append(
                        TestCase(stdin=example["input"], stdout=example["output"])
                    )
        if benchmark.get("note"):
            question += "\n\nNote\n\n" + benchmark["note"]

    return question, tests


def build_question_id_to_problem_mapping(
    cache_path: Path = Path("question_id_to_problem.json"),
    num_examples: Optional[int] = None,
) -> Tuple[Dict[str, Problem], List[Problem]]:
    """
    Build mapping from question_id to Problem.
    Load from cache if it exists, otherwise build it and save to cache.

    Args:
        cache_path: Path to cache file
        num_examples: Optional limit on number of examples to process (for testing)

    Returns:
        Tuple of (question_id_to_problem dict, list of all problems)
    """

    # Check if cache exists
    if cache_path.exists():
        print(f"Loading cached mapping from {cache_path}")
        with open(cache_path, "r") as f:
            cached_data = json.load(f)

            # Reconstruct Problem instances from JSON
            problems = [Problem(**prob_data) for prob_data in cached_data["problems"]]

            # Rebuild the question_id to Problem mapping
            question_id_to_problem = {}
            for prob_idx, question_ids in cached_data[
                "question_id_to_problem_idx"
            ].items():
                prob_idx = int(prob_idx)
                for qid in question_ids:
                    question_id_to_problem[qid] = problems[prob_idx]

            return question_id_to_problem, problems

    print("Building question_id to Problem mapping...")

    # First, collect all unique question_ids to identify unique problems
    question_id_to_problem_key: Dict[str, Tuple[str, str, int]] = {}
    problem_key_to_question_ids: Dict[Tuple[str, str, int], List[str]] = {}

    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

    # Scan all samples to identify unique problems
    examples_processed = 0
    for lang in ["python", "cpp"]:
        ocr2_ds = ocr2_dataset[lang]  # type: ignore
        for ocr2_ds_item in tqdm(ocr2_ds, desc=f"Scanning {lang} for unique problems"):
            if num_examples is not None and examples_processed >= num_examples:
                break
            item: Dict[str, Any] = ocr2_ds_item  # type: ignore

            ds_name: str = item["dataset"]
            ds_split: str = item["split"]
            ds_index: int = int(item["index"])
            question_id: str = item.get("question_id", "")

            problem_key = (ds_name, ds_split, ds_index)

            if question_id:
                question_id_to_problem_key[question_id] = problem_key

                if problem_key not in problem_key_to_question_ids:
                    problem_key_to_question_ids[problem_key] = []
                if question_id not in problem_key_to_question_ids[problem_key]:
                    problem_key_to_question_ids[problem_key].append(question_id)

            examples_processed += 1

        if num_examples is not None and examples_processed >= num_examples:
            break

    # Create Problem instances
    print(f"\nCreating {len(problem_key_to_question_ids)} unique problems...")
    problems: List[Problem] = []
    question_id_to_problem: Dict[str, Problem] = {}

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

        # Create Problem with empty samples list (will be filled later)
        problem = Problem(
            scenario_config=scenario_config,
            dataset=ds_name,
            split=ds_split,
            index=ds_index,
            samples=[],
        )

        problems.append(problem)

        # Map all question_ids for this problem to the Problem instance
        for qid in question_ids:
            question_id_to_problem[qid] = problem

    # Save to cache
    print(f"\nSaving mapping to cache at {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a mapping from question_id to problem index for more efficient storage
    question_id_to_problem_idx = {}
    problem_to_idx = {id(prob): idx for idx, prob in enumerate(problems)}

    for qid, prob in question_id_to_problem.items():
        prob_idx = problem_to_idx[id(prob)]
        if prob_idx not in question_id_to_problem_idx:
            question_id_to_problem_idx[prob_idx] = []
        question_id_to_problem_idx[prob_idx].append(qid)

    # Convert to JSON-serializable format
    cache_data = {
        "problems": [prob.model_dump() for prob in problems],
        "question_id_to_problem_idx": {
            str(k): v for k, v in question_id_to_problem_idx.items()
        },
    }

    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    return question_id_to_problem, problems


def append_samples_to_problems(
    question_id_to_problem: Dict[str, Problem], num_examples: Optional[int] = None
) -> None:
    """
    Append all OCR2 samples to their corresponding problems using the question_id mapping.

    Args:
        question_id_to_problem: Mapping from question_id to Problem instance
        num_examples: Optional limit on number of examples to process (for testing)
    """

    print("\nAppending samples to problems...")
    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

    samples_added = 0
    samples_skipped = 0
    examples_processed = 0

    for lang in ["python", "cpp"]:
        ocr2_ds = ocr2_dataset[lang]  # type: ignore
        for ocr2_ds_item in tqdm(ocr2_ds, desc=f"Processing {lang} samples"):
            if num_examples is not None and examples_processed >= num_examples:
                break
            item: Dict[str, Any] = ocr2_ds_item  # type: ignore

            question_id = item.get("question_id", "")

            if question_id and question_id in question_id_to_problem:
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

                # Add sample to the corresponding problem
                question_id_to_problem[question_id].samples.append(sample)
                samples_added += 1
            else:
                if question_id:
                    print(
                        f"Warning: Sample with question_id '{question_id}' has no corresponding problem"
                    )
                samples_skipped += 1

            examples_processed += 1

        if num_examples is not None and examples_processed >= num_examples:
            break

    print(
        f"\nAdded {samples_added} samples to problems, skipped {samples_skipped} samples"
    )


def convert_ocr2_to_problems(
    cache_path: Path = Path("question_id_to_problem.json"),
    num_examples: Optional[int] = None,
) -> List[Problem]:
    """
    Convert OpenCodeReasoning2 dataset items to Problem instances.

    Args:
        cache_path: Path to cache file for question_id to Problem mapping
        num_examples: Optional limit on number of examples to process (for testing)

    Returns:
        List of Problem instances with samples attached
    """

    # Step 1: Build or load the question_id to Problem mapping
    question_id_to_problem, problems = build_question_id_to_problem_mapping(
        cache_path, num_examples
    )

    # Step 2: Append all samples to their corresponding problems
    append_samples_to_problems(question_id_to_problem, num_examples)

    return problems


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert OCR2 dataset to Problem instances")
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=None,
        help="Limit number of examples to process (for testing)"
    )
    parser.add_argument(
        "--cache-prefix",
        type=str,
        default="question_id_to_problem",
        help="Prefix for cache file name (default: question_id_to_problem)"
    )
    
    args = parser.parse_args()
    
    # Construct cache path based on prefix and num_examples
    if args.num_examples:
        # Use a different cache file for limited runs
        cache_path = Path(f"{args.cache_prefix}_{args.num_examples}.json")
    else:
        cache_path = Path(f"{args.cache_prefix}.json")
    
    if args.num_examples:
        print(f"Processing limited to {args.num_examples} examples")
        problems = convert_ocr2_to_problems(cache_path, args.num_examples)
    else:
        print("Processing all examples")
        problems = convert_ocr2_to_problems(cache_path)
    print(f"\nConverted {len(problems)} unique problems")

    # Count total samples
    total_samples = sum(len(p.samples) for p in problems)
    print(f"Total samples across all problems: {total_samples}")

    if problems:
        print(f"\nFirst problem example:")
        print(f"Dataset: {problems[0].dataset}")
        print(f"Prompt: {problems[0].scenario_config.prompt[:200]}...")
        print(f"Number of tests: {len(problems[0].scenario_config.tests)}")
        print(f"Number of samples: {len(problems[0].samples)}")
        if problems[0].samples:
            print(
                f"First sample has R1 generation: {bool(problems[0].samples[0].r1_generation)}"
            )
