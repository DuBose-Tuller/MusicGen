#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it and try again."
    exit 1
fi

# Default values
mix_ratios=(0.5)  # Mix ratios to test
output_dataset="mixed_experiment"  # Name for the new dataset
segment_length=""
stride_length=""

# Parse command line arguments
while getopts ":o:r:s:t:" opt; do
  case $opt in
    o) output_dataset="$OPTARG"     # Output dataset name
    ;;
    r) IFS=',' read -ra mix_ratios <<< "$OPTARG"  # Comma-separated mix ratios
    ;;
    s) segment_length="$OPTARG"
    ;;
    t) stride_length="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

# Shift past the options
shift $((OPTIND-1))

# Get positional arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 [options] performance_dataset noise_dataset"
    echo "Options:"
    echo "  -o output_dataset    Name for the output dataset"
    echo "  -r mix_ratios       Comma-separated list of mix ratios (e.g., 0.2,0.5,0.8)"
    echo "  -s segment_length   Segment length in seconds"
    echo "  -t stride_length    Stride length in seconds"
    echo "Example: $0 -o mixed_experiment -r 0.2,0.5,0.8 performances noise"
    exit 1
fi

perf_dataset="$1"
noise_dataset="$2"

# Determine input and output directory structure based on segment/stride
if [ -n "$segment_length" ] && [ -n "$stride_length" ]; then
    subdir="s${segment_length}-t${stride_length}"
else
    subdir="raw"
fi

# Set up directory paths
perf_dir="../data/${perf_dataset}/${subdir}"
noise_dir="../data/${noise_dataset}/${subdir}"
output_dir="../data/${output_dataset}/${subdir}"

# Validate inputs
if [ ! -d "$perf_dir" ]; then
    echo "Error: Performance directory not found: $perf_dir"
    exit 1
fi

if [ ! -d "$noise_dir" ]; then
    echo "Error: Noise directory not found: $noise_dir"
    exit 1
fi

# Create output directory
mkdir -p "$output_dir"

# Process each performance file
for perf_file in "$perf_dir"/*.wav; do
    perf_name=$(basename "$perf_file" .wav)
    
    # Process each noise file
    for noise_file in "$noise_dir"/*.wav; do
        noise_name=$(basename "$noise_file" .wav)
        
        # Create mixed versions at different ratios
        for ratio in "${mix_ratios[@]}"; do
            output_name="${perf_name}_noise${noise_name}_mix${ratio}.wav"
            output_path="${output_dir}/${output_name}"
            
            # Skip if file already exists
            if [ -f "$output_path" ]; then
                echo "Skipping existing file: $output_path"
                continue
            fi
            
            # Calculate volume adjustments for mixing
            perf_vol=$(echo "1 - $ratio" | bc -l)
            noise_vol=$ratio
            
            # Mix audio using ffmpeg
            ffmpeg -i "$perf_file" -i "$noise_file" \
                -filter_complex "[0:a]aformat=sample_fmts=fltp:sample_rates=32000:channel_layouts=mono[a1]; \
                               [1:a]aformat=sample_fmts=fltp:sample_rates=32000:channel_layouts=mono,aloop=0:1000:0[a2]; \
                               [a1][a2]amix=inputs=2:weights=${perf_vol} ${noise_vol}:duration=first[aout]" \
                -map "[aout]" \
                -ar 32000 -ac 1 "$output_path" -y >/dev/null
            
            echo "Created: $output_path"
        done
    done
done

echo "Processing complete. Dataset saved in: $output_dir"

# Create metadata file with proper JSON formatting
metadata_file="${output_dir}/metadata.json"
{
    echo "{"
    echo "  \"dataset_info\": {"
    echo "    \"name\": \"${output_dataset}\","
    echo "    \"performance_source\": \"${perf_dataset}\","
    echo "    \"noise_source\": \"${noise_dataset}\","
    echo "    \"mix_ratios\": [$(printf "%.1f," "${mix_ratios[@]}" | sed 's/,$/')],"
    echo "    \"segment_length\": \"${segment_length}\","
    echo "    \"stride_length\": \"${stride_length}\""
    echo "  }"
    echo "}"
} > "$metadata_file"

echo "Metadata saved to: $metadata_file"