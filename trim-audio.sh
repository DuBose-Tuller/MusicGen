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

# Function to process a single directory
process_directory() {
    local dir="$1"
    local base_dir=$(dirname "$dir")
    local dir_name=$(basename "$dir")
    local output_dir="${base_dir}/${dir_name}-trimmed-s${segment_length}-t${stride_length}"
    
    # Skip if the directory name contains 'trimmed'
    if [[ $dir_name == *"trimmed"* ]]; then
        echo "Skipping already processed directory: $dir"
        return
    fi
    
    mkdir -p "$output_dir"
    
    for file in "$dir"/*.wav; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            
            # Skip if the filename contains 'trimmed'
            if [[ $filename == *"trimmed"* ]]; then
                echo "Skipping already trimmed file: $file"
                continue
            fi
            
            duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null)
            
            for start in $(seq 0 $stride_length $(echo "$duration - $segment_length" | bc)); do
                output_file="${output_dir}/${filename%.*}_trimmed_${start}.wav"
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

# Process the main directory and its subdirectories
for dir in "$input_dir"/*; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        process_directory "$dir"
    fi
done

echo "Processing complete."