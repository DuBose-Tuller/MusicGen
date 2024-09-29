#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it and try again."
    exit 1
fi

# Default values
segment_length=5
stride_length=5

# Parse command line arguments
while getopts ":s:t:" opt; do
  case $opt in
    s) segment_length="$OPTARG"
    ;;
    t) stride_length="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

# Shift the parsed options out of the argument list
shift $((OPTIND -1))

# Check if a directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [-s segment_length] [-t stride_length] <directory>"
    exit 1
fi

input_dir="$1"

# Function to process a single dataset directory
process_dataset() {
    local dataset_dir="$1"
    local raw_dir="${dataset_dir}/raw"
    local output_dir="${dataset_dir}/s${segment_length}-t${stride_length}"
    
    # Check if raw directory exists
    if [ ! -d "$raw_dir" ]; then
        echo "No 'raw' directory found in ${dataset_dir}. Skipping."
        return
    }
    
    # Check if output directory already exists
    if [ -d "$output_dir" ]; then
        echo "Output directory ${output_dir} already exists. Skipping processing for this dataset."
        return
    fi
    
    mkdir -p "$output_dir"
    
    for file in "$raw_dir"/*.wav; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null)
            
            for start in $(seq 0 $stride_length $(echo "$duration - $segment_length" | bc)); do
                output_file="${output_dir}/${filename%.*}_${start}.wav"
                if [ ! -f "$output_file" ]; then
                    ffmpeg -i "$file" -ss $start -t $segment_length -c copy "$output_file" -y >/dev/null 2>&1
                    echo "Processed: $file -> $output_file"
                else
                    echo "Skipping existing file: $output_file"
                fi
            done
        fi
    done
}

# Process each dataset directory
for dataset_dir in "$input_dir"/*; do
    if [ -d "$dataset_dir" ]; then
        echo "Processing dataset: $dataset_dir"
        process_dataset "$dataset_dir"
    fi
done

echo "Processing complete."