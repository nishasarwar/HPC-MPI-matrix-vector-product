#!/bin/bash

# Configuration
ROWS=10000
COLS=3000001
SAMPLES=5
EXEC=./matvec
OUTPUT="strong_scaling_results.txt"
MAX_RANKS=256

# Prepare output text file
echo -e "Ranks\tAvg_Compute_μs\tAvg_Communicate_μs" > $OUTPUT

# Sweep over powers of 2 up to MAX_RANKS
for RANKS in 1 2 4 8 16 32 64 128 256; do
    echo "Running with $RANKS ranks..."

    # Run the executable and capture output
    RESULT=$(srun -n $RANKS $EXEC $ROWS $COLS $SAMPLES)

    # Extract average compute and communication times
    AVG_COMPUTE=$(echo "$RESULT" | grep "average compute μs" | awk '{print $4}')
    AVG_COMM=$(echo "$RESULT" | grep "average communicate μs" | awk '{print $4}')

    # Append to output file in tabular format
    printf "%-6s\t%-15s\t%-18s\n" "$RANKS" "$AVG_COMPUTE" "$AVG_COMM" >> $OUTPUT
done

echo "Results saved to $OUTPUT"
