# Competitive Coding Pipeline

Processes OpenCodeReasoning2 dataset into partitioned problem sets with SFT/RL training formats.

## Quick Start

Run all steps:
```bash
# Step 1: Create blank problems
uv run python src/comp_coding/step1_create_blank_problems.py --n-workers 128

# Step 2: Append and filter samples (combined - avoids storing all samples)
uv run python src/comp_coding/step2_append_and_filter.py --n-workers 128

# Step 3: Partition into 4 sets and create training splits
uv run python src/comp_coding/step3_partition_problems.py --create-training-splits
```

## Pipeline Steps

1. **step1_create_blank_problems.py** - Maps OCR2 to unique problems
   - Output: `data/problems_step1_blank.json`, `data/problems_step1_mapping.json`

2. **step2_append_and_filter.py** - Appends samples and filters to top 4 per problem
   - Combines appending and filtering to avoid storing all samples
   - Filters by pass_rate (highest first), then length (shortest first)
   - Output: `data/problems_step2_filtered.json`

3. **step3_partition_problems.py** - Creates 4 equal partitions with optional training splits
   - Partitions: `data/problems_step3_partition_{1-4}.json`
   - Training splits (when --create-training-splits):
     - 75% SFT / 25% RL
     - 50% SFT / 50% RL  
     - 25% SFT / 75% RL
     - Luffy: 100% SFT AND RL (off-policy IS)
   - Training files: `data/training_splits/partition_{1-4}_{sft|rl}_{split}.jsonl`

## Options

```bash
--num-examples N          # Limit to N examples (testing)
--n-workers N            # Parallel workers (default: CPU count, steps 1-2 only)
--force-recreate         # Ignore cache
--random-seed N          # Partition seed (default: 42)
--create-training-splits # Generate SFT/RL training formats (step3)
```

## Test with 100 examples
```bash
uv run python src/comp_coding/step1_create_blank_problems.py --num-examples 100
uv run python src/comp_coding/step2_append_and_filter.py --num-examples 100
uv run python src/comp_coding/step3_partition_problems.py --num-examples 100 --create-training-splits
```
