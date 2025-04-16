#!/bin/bash

# Configuration
COLS=300
SAMPLES=16
EXEC=./matvec
OUTPUT="q5_scaling_results.txt"

# Header for the output
echo "rows,avg_communicate_us" > $OUTPUT

# Loop over rows from 500 to 10000 in steps of 500
for ROWS in $(seq 500 500 10000); do
    echo "Running with $ROWS rows..."

    # Run the command
    RESULT=$(srun -n 256 $EXEC $ROWS $COLS $SAMPLES)

    # Extract average communication time
    AVG_COMM=$(echo "$RESULT" | grep "average communicate μs" | awk '{print $4}')

    # Append to file
    echo "$ROWS,$AVG_COMM" >> $OUTPUT
done

echo "Results saved to $OUTPUT"
