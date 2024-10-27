#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it and try again."
    exit 1
fi

# Default values
segment_length=15
stride_length=15
sample_rate=32000

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
    local raw_dir="${dataset_dir}"
    
    # If the input directory has a 'raw' subdirectory, use that
    if [ -d "${dataset_dir}/raw" ]; then
        raw_dir="${dataset_dir}/raw"
    fi
    
    # Determine the base directory for output
    local base_dir=$(dirname "$dataset_dir")
    local dataset_name=$(basename "$dataset_dir")
    if [ "$raw_dir" = "${dataset_dir}/raw" ]; then
        dataset_name=$(basename "$dataset_dir")
    else
        # If we're processing a raw directory directly, go up one level
        base_dir=$(dirname "$base_dir")
        dataset_name=$(basename $(dirname "$raw_dir"))
    fi
    
    local output_dir="${base_dir}/${dataset_name}/s${segment_length}-t${stride_length}"
    
    echo "Processing dataset directory: $raw_dir"
    echo "Output directory will be: $output_dir"
    
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
            
            # Check if duration is valid
            if [ -z "$duration" ]; then
                echo "Warning: Could not determine duration for $file. Skipping."
                continue
            fi
            
            # Remove any carriage returns from duration
            duration=$(echo "$duration" | tr -d '\r')
            
            # If file is shorter than segment length, just copy it with resampling
            if (( ${duration%.*} < $segment_length )); then
                echo "File $file is shorter than segment length. Copying with resampling."
                output_file="${output_dir}/${filename}"
                if ffmpeg -i "$file" -ar $sample_rate -y "$output_file" >/dev/null 2>&1; then
                    echo "Processed: $file -> $output_file"
                else
                    echo "Error processing $file. Skipping."
                fi
                continue
            fi
            
            # Process longer files into segments
            current_pos=0
            while (( current_pos + segment_length <= ${duration%.*} )); do
                output_file="${output_dir}/${filename%.*}_${current_pos}.wav"
                
                if [ ! -f "$output_file" ]; then
                    if ffmpeg -i "$file" -ss $current_pos -t $segment_length -ar $sample_rate -y "$output_file" >/dev/null 2>&1; then
                        echo "Processed: $file -> $output_file"
                    else
                        echo "Error processing $file at position $current_pos. Skipping."
                        rm -f "$output_file" # Clean up potentially partial output file
                    fi
                else
                    echo "Skipping existing file: $output_file"
                fi
                
                current_pos=$((current_pos + stride_length))
            done
        fi
    done
}

# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Error: Directory $input_dir does not exist"
    exit 1
fi

# If the input directory contains a 'raw' subdirectory or is itself a raw directory,
# process it directly. Otherwise, look for dataset subdirectories.
if [ -d "${input_dir}/raw" ] || [[ $(basename "$input_dir") == "raw" ]]; then
    process_dataset "$input_dir"
else
    # Check if we're already in a dataset directory
    parent_dir=$(dirname "$input_dir")
    if [ -d "${input_dir}/raw" ]; then
        process_dataset "$input_dir"
    else
        # Process each dataset directory
        for dataset_dir in "$input_dir"/*; do
            if [ -d "$dataset_dir" ]; then
                process_dataset "$dataset_dir"
            fi
        done
    fi
fi

echo "Processing complete."