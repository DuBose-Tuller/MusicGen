#!/bin/bash

# add_noise.sh - Add various levels of noise to audio files while preserving directory structure
#
# This script can process audio files by adding different levels of noise using FFmpeg.
# It works with both individual datasets and the entire data directory structure.
# It preserves the original directory structure and creates new directories for each noise level.
#
# Directory Structure Examples:
#   data/                          # Root data directory
#   ├── dataset1/                  # Dataset directory
#   │   ├── raw/                  # Original files
#   │   ├── s15-t15/             # Segmented files (if using -s and -t)
#   │   ├── noise25_raw/         # 25% noise added to raw files
#   │   └── noise25_s15-t15/     # 25% noise added to segmented files
#   └── dataset2/                  # Another dataset
#       └── ...
#
# Usage Examples:
#   1. Process entire data directory with default settings:
#      ./add_noise.sh ../data
#
#   2. Process specific dataset with default settings:
#      ./add_noise.sh ../data/acpas
#
#   3. Process specific folder without segmentation:
#      ./add_noise.sh ../data/acpas/raw
#
#   4. Process with custom segment length and stride:
#      ./add_noise.sh -s 15 -t 15 ../data
#
#   5. Process with custom noise levels:
#      ./add_noise.sh -n 0.10,0.30,0.60 ../data
#
# Options:
#   -s <length>     Segment length in seconds (must be used with -t)
#   -t <length>     Stride length in seconds (must be used with -s)
#   -n <levels>     Comma-separated list of noise levels (0.0 to 1.0)
#                   Default: 0.25,0.50,0.75
#
# Notes:
#   - When processing a raw/ directory directly, -s and -t options are ignored
#   - Original files are preserved; new files are created in noise{XX}/ directories
#   - Script requires FFmpeg to be installed

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it and try again."
    exit 1
fi

# Default values
segment_length=""
stride_length=""
noise_levels=(0.25 0.50 0.75)  # Default noise levels (25%, 50%, 75%)

# Parse command line arguments
while getopts ":s:t:n:" opt; do
  case $opt in
    s) segment_length="$OPTARG"
    ;;
    t) stride_length="$OPTARG"
    ;;
    n) IFS=',' read -ra noise_levels <<< "$OPTARG"  # Allow custom noise levels
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
    echo "Usage: $0 [-s segment_length] [-t stride_length] [-n noise_levels] <directory>"
    echo "Example: $0 -s 15 -t 15 -n 0.25,0.50,0.75 ../data"
    exit 1
fi

# Convert input path to absolute path
input_dir=$(realpath "$1")

# Function to process a single directory with noise
process_directory_with_noise() {
    local source_dir="$1"
    local noise_level="$2"
    local base_dir=$(dirname "$source_dir")
    local dir_name=$(basename "$source_dir")
    
    # Convert noise level to percentage for directory naming
    local noise_percent=$(echo "$noise_level * 100" | bc)
    local output_dir="${base_dir}/noise${noise_percent%.*}_${dir_name}"
    
    # Check if output directory already exists
    if [ -d "$output_dir" ]; then
        echo "Output directory ${output_dir} already exists. Skipping processing for this noise level."
        return
    fi
    
    mkdir -p "$output_dir"
    
    for file in "$source_dir"/*.wav; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            output_file="${output_dir}/${filename}"
            
            if [ ! -f "$output_file" ]; then
                # Add noise using ffmpeg
                ffmpeg -i "$file" -filter_complex \
                    "[0:a]asplit=2[a][b];
                     aevalsrc=random(0)*2-1:n=2:s=${sample_rate}[noise];
                     [noise]volume=${noise_level}[noise2];
                     [a][noise2]amix=inputs=2:duration=first[mixed];
                     [mixed]volume=2[normalized];
                     [normalized][b]amix=inputs=2:duration=first:weights=5 1" \
                    "$output_file" -y >/dev/null 2>&1
                
                echo "Processed: $file -> $output_file (${noise_percent}% noise)"
            else
                echo "Skipping existing file: $output_file"
            fi
        fi
    done
}

# Function to determine if a path is a dataset directory
is_dataset_dir() {
    local dir="$1"
    # Check if directory contains raw/ or s*-t*/ subdirectories
    if [ -d "${dir}/raw" ] || ls -d "${dir}"/s*-t*/ >/dev/null 2>&1; then
        return 0  # True
    fi
    return 1  # False
}

# Main processing logic
if [ -d "$input_dir" ]; then
    if [[ "$input_dir" == */raw ]]; then
        # Direct processing of a raw directory
        echo "Processing raw directory: $input_dir"
        for noise_level in "${noise_levels[@]}"; do
            process_directory_with_noise "$input_dir" "$noise_level"
        done
    elif [[ "$input_dir" =~ /s[0-9]+-t[0-9]+$ ]]; then
        # Direct processing of a segmented directory
        echo "Processing segmented directory: $input_dir"
        for noise_level in "${noise_levels[@]}"; do
            process_directory_with_noise "$input_dir" "$noise_level"
        done
    else
        # Process each dataset directory
        if is_dataset_dir "$input_dir"; then
            # Single dataset directory
            echo "Processing dataset: $input_dir"
            if [[ -n $stride_length && -n $segment_length ]]; then
                source_dir="${input_dir}/s${segment_length}-t${stride_length}"
            else
                source_dir="${input_dir}/raw"
            fi
            
            if [ -d "$source_dir" ]; then
                for noise_level in "${noise_levels[@]}"; do
                    process_directory_with_noise "$source_dir" "$noise_level"
                done
            else
                echo "Source directory not found: $source_dir"
            fi
        else
            # Multiple datasets
            echo "Processing all datasets in: $input_dir"
            for dataset_dir in "$input_dir"/*; do
                if [ -d "$dataset_dir" ] && is_dataset_dir "$dataset_dir"; then
                    echo "Processing dataset: $dataset_dir"
                    if [[ -n $stride_length && -n $segment_length ]]; then
                        source_dir="${dataset_dir}/s${segment_length}-t${stride_length}"
                    else
                        source_dir="${dataset_dir}/raw"
                    fi
                    
                    if [ -d "$source_dir" ]; then
                        for noise_level in "${noise_levels[@]}"; do
                            process_directory_with_noise "$source_dir" "$noise_level"
                        done
                    else
                        echo "Source directory not found: $source_dir"
                    fi
                fi
            done
        fi
    fi
else
    echo "Error: Directory not found: $input_dir"
    exit 1
fi

echo "Processing complete."