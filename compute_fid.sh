#!/bin/bash

# compute_fid.sh
# This script computes FID scores for epochs 111 to 130
# and saves the results to corresponding text files.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define directories
REAL_IMAGES_DIR="real_images/"
SAMPLES_DIR_BASE="samples_v3/"
RESULTS_DIR="results/"

# Create the results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Loop through epochs 111 to 130
for epoch in {111..130}
do
    SAMPLES_DIR="${SAMPLES_DIR_BASE}${epoch}/"
    RESULT_FILE="${RESULTS_DIR}${epoch}.txt"
    
    # Check if the samples directory exists
    if [ -d "$SAMPLES_DIR" ]; then
        echo "Calculating FID for epoch $epoch..."
        
        # Calculate FID and save the output to the result file
        python -m pytorch_fid "$REAL_IMAGES_DIR" "$SAMPLES_DIR" > "$RESULT_FILE"
        
        echo "FID for epoch $epoch saved to $RESULT_FILE"
    else
        echo "Warning: Samples directory $SAMPLES_DIR does not exist. Skipping epoch $epoch."
    fi
done

echo "FID calculation completed for epochs 111 to 130."
