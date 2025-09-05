#!/bin/bash
# Run the complete OCR2 processing pipeline

set -e  # Exit on error

# Parse command line arguments
NUM_EXAMPLES=""
N_WORKERS="128"
MAX_SAMPLES="4"
FORCE_RECREATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-examples)
            NUM_EXAMPLES="--num-examples $2"
            shift 2
            ;;
        --n-workers)
            N_WORKERS="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --force-recreate)
            FORCE_RECREATE="--force-recreate"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-examples N     Limit to N examples (for testing)"
            echo "  --n-workers N        Number of parallel workers (default: 128)"
            echo "  --max-samples N      Max samples per problem (default: 4)"
            echo "  --force-recreate     Force recreation, ignore cache"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Running Full OCR2 Processing Pipeline"
echo "=========================================="
echo ""

if [ -n "$NUM_EXAMPLES" ]; then
    echo "Processing limited examples: $NUM_EXAMPLES"
else
    echo "Processing ALL examples from OCR2 dataset"
fi
echo "Workers: $N_WORKERS"
echo "Max samples per problem: $MAX_SAMPLES"
if [ -n "$FORCE_RECREATE" ]; then
    echo "Force recreate: enabled"
fi
echo ""

# Step 1: Create blank problems and mapping
echo "Step 1: Creating blank problems and mapping..."
echo "------------------------------------------"
uv run python -m comp_coding.step1_create_blank_problems \
    --n-workers $N_WORKERS \
    $NUM_EXAMPLES \
    $FORCE_RECREATE

echo ""
echo "Step 1 complete!"
echo ""

# Step 2: Append and filter samples
echo "Step 2: Appending and filtering samples..."
echo "Keeping top $MAX_SAMPLES samples per problem"
echo "------------------------------------------"
uv run python -m comp_coding.step2_append_and_filter \
    --n-workers $N_WORKERS \
    --max-samples $MAX_SAMPLES \
    $NUM_EXAMPLES \
    $FORCE_RECREATE

echo ""
echo "Step 2 complete!"
echo ""

# Step 3: Partition into 4 sets and create training splits
echo "Step 3: Creating partitions and aggregated training splits..."
echo "Creating 4 partitions with SFT/RL splits"
echo "------------------------------------------"
uv run python -m comp_coding.step3_partition_problems \
    --num-partitions 4 \
    --create-training-splits \
    $NUM_EXAMPLES \
    $FORCE_RECREATE

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""

# Display output files
echo "Output files created:"
if [ -n "$NUM_EXAMPLES" ]; then
    PREFIX="problems_${NUM_EXAMPLES##* }"
else
    PREFIX="problems"
fi

echo ""
echo "Main files:"
echo "  - data/${PREFIX}_step1_blank.json"
echo "  - data/${PREFIX}_step1_mapping.json"
echo "  - data/${PREFIX}_step2_filtered.json"
echo "  - data/${PREFIX}_step2_filtered_stats.json"
echo "  - data/${PREFIX}_step3_partition_1.json"
echo "  - data/${PREFIX}_step3_partition_2.json"
echo "  - data/${PREFIX}_step3_partition_3.json"
echo "  - data/${PREFIX}_step3_partition_4.json"
echo "  - data/${PREFIX}_step3_partition_metadata.json"

echo ""
echo "Training splits (aggregated across partitions):"
echo "  - data/training_splits/split_75_25_sft.jsonl"
echo "  - data/training_splits/split_75_25_rl.jsonl"
echo "  - data/training_splits/split_50_50_sft.jsonl"
echo "  - data/training_splits/split_50_50_rl.jsonl"
echo "  - data/training_splits/split_25_75_sft.jsonl"
echo "  - data/training_splits/split_25_75_rl.jsonl"
echo "  - data/training_splits/split_luffy_sft.jsonl"
echo "  - data/training_splits/split_luffy_rl.jsonl"
echo "  - data/training_splits/training_splits_stats.json"
echo ""