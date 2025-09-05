# Competitive Coding Pipeline

Processes OpenCodeReasoning2 dataset into partitioned problem sets with SFT/RL training formats.

## Dataset Statistics

From processing 1.4M OpenCodeReasoning2 Python examples:
- **34,125 unique problems** identified
- **136,436 total samples** (4 samples per problem max)
- **Average 29.6 tests per problem** (range: 0 to 1,440 tests)

## Key Decisions

1. **Sample limiting to 4 per problem**: Keeps top 4 samples ranked by pass_rate (highest first), then solution length (shortest first)
2. **Test limiting to 16 per problem**: Caps maximum tests at 16 for RL formats to reduce file sizes (~35% reduction)

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
   - **Filtering criteria**: Keeps top 4 samples per problem
     - Sorts by pass_rate (highest first)
     - Then by solution length (shortest first)
   - Output: `data/problems_step2_filtered.json`

3. **step3_partition_problems.py** - Creates 4 equal partitions with optional training splits
   - Partitions: `data/problems_step3_partition_{1-4}.json`
   - **Test limiting**: RL problems limited to 16 tests maximum (saves ~35% space)
   - Training splits (when --create-training-splits):
     - 100% SFT / 0% RL (all partitions for SFT)
     - 75% SFT / 25% RL (partitions 1-3 for SFT, partition 4 for RL)
     - 50% SFT / 50% RL (partitions 1-2 for SFT, partitions 3-4 for RL) 
     - 25% SFT / 75% RL (partition 1 for SFT, partitions 2-4 for RL)
     - 0% SFT / 100% RL (all partitions for RL)
     - Luffy: RLProblemWithCompletions format (includes r1_generation completions for off-policy IS)
   - Training files: `data/training_splits/split_{ratio}_sft.jsonl`, `split_{ratio}_rl.jsonl`, `split_luffy.jsonl`

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
