# Competitive Coding Pipeline

Processes OpenCodeReasoning2 dataset into partitioned problem sets.

## Quick Start

Run all steps:
```bash
# Step 1: Create blank problems
uv run python src/comp_coding/step1_create_blank_problems.py --n-workers 128

# Step 2: Append and filter samples (combined - avoids storing all samples)
uv run python src/comp_coding/step2_append_and_filter.py --n-workers 128

# Step 3: Partition into 4 sets
uv run python src/comp_coding/step3_partition_problems.py
```

## Pipeline Steps

1. **step1_create_blank_problems.py** - Maps OCR2 to unique problems
   - Output: `data/problems_step1_blank.json`, `data/problems_step1_mapping.json`

2. **step2_append_and_filter.py** - Appends samples and filters to top 4 per problem
   - Combines appending and filtering to avoid storing all samples
   - Filters by pass_rate (highest first), then length (shortest first)
   - Output: `data/problems_step2_filtered.json`

3. **step3_partition_problems.py** - Creates 4 equal partitions
   - Output: `data/problems_step3_partition_{1-4}.json`

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
