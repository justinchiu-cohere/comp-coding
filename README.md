# Competitive Coding Pipeline

Processes OpenCodeReasoning2 dataset into partitioned problem sets.

## Quick Start

Run all steps:
```bash
# Step 1: Create blank problems
uv run python src/comp_coding/step1_create_blank_problems.py

# Step 2: Append samples  
uv run python src/comp_coding/step2_append_all_samples.py

# Step 3: Filter to top 4 samples
uv run python src/comp_coding/step3_filter_samples.py

# Step 4: Partition into 4 sets
uv run python src/comp_coding/step4_partition_problems.py
```

## Pipeline Steps

1. **step1_create_blank_problems.py** - Maps OCR2 to unique problems
   - Output: `data/problems_step1_blank.json`, `data/problems_step1_mapping.json`

2. **step2_append_all_samples.py** - Adds all R1 samples
   - Output: `data/problems_step2_all_samples.json`

3. **step3_filter_samples.py** - Keeps top 4 samples (by pass_rate, then length)
   - Output: `data/problems_step3_filtered.json`

4. **step4_partition_problems.py** - Creates 4 equal partitions
   - Output: `data/problems_step4_partition_{1-4}.json`

## Options

```bash
--num-examples N     # Limit to N examples (testing)
--n-workers N        # Parallel workers (default: CPU count)
--force-recreate     # Ignore cache
--random-seed N      # Partition seed (default: 42)
```

## Test with 100 examples
```bash
uv run python src/comp_coding/step1_create_blank_problems.py --num-examples 100
uv run python src/comp_coding/step2_append_all_samples.py --num-examples 100
uv run python src/comp_coding/step3_filter_samples.py --num-examples 100
uv run python src/comp_coding/step4_partition_problems.py --num-examples 100
```