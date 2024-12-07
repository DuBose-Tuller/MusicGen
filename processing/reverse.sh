#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it and try again."
    exit 1
fi

# Check if a directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

input_dir="$1"

# Function to process a single dataset directory
process_dataset() {
    echo "$1"
    local dataset_dir="$1"
    local raw_dir="${dataset_dir}/raw"
    local output_dir="${dataset_dir}/reversed"
    
    # Check if raw directory exists
    if [ ! -d "$raw_dir" ]; then
        echo "No 'raw' directory found in ${dataset_dir}. Skipping."
        return
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Process each WAV file
    for file in "$raw_dir"/*.wav; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            output_file="${output_dir}/${filename%.*}_reversed.wav"
            
            if [ ! -f "$output_file" ]; then
                # Reverse the audio file
                ffmpeg -i "$file" -af areverse "$output_file" -y >/dev/null 2>&1
                echo "Processed: $file -> $output_file"
            else
                echo "Skipping existing file: $output_file"
            fi
        fi
    done
}

# Process each dataset directory
# for dataset_dir in "$input_dir"/*; do
#     if [ -d "$dataset_dir" ]; then
#         echo "Processing dataset: $dataset_dir"
#         process_dataset "$dataset_dir"
#     fi
# done

echo "Processing dataset: $1"
process_dataset $1

echo "Processing complete."