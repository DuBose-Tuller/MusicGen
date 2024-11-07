#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it and try again."
    exit 1
fi

# Default values
mix_ratios=(0.2 0.5 0.8)  # Mix ratios to test
dataset_name="mixed_experiment"  # Name for the new dataset

# Parse command line arguments
while getopts ":p:n:o:r:" opt; do
  case $opt in
    p) performance_dir="$OPTARG"  # Directory containing performance audio
    ;;
    n) noise_dir="$OPTARG"        # Directory containing noise audio
    ;;
    o) dataset_name="$OPTARG"     # Output dataset name
    ;;
    r) IFS=',' read -ra mix_ratios <<< "$OPTARG"  # Comma-separated mix ratios
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

# Validate inputs
if [ -z "$performance_dir" ] || [ -z "$noise_dir" ]; then
    echo "Usage: $0 -p performance_dir -n noise_dir [-o output_dataset_name] [-r mix_ratios]"
    echo "Example: $0 -p data/performances/raw -n data/noise/raw -o mixed_experiment -r 0.2,0.5,0.8"
    exit 1
fi

# Create output directory structure
output_dir="data/${dataset_name}/raw"
mkdir -p "$output_dir"

# Function to normalize audio levels
normalize_audio() {
    local input="$1"
    local output="$2"
    ffmpeg -i "$input" -filter:a loudnorm -ar 32000 -ac 1 "$output" -y
}

# Process each performance file
for perf_file in "$performance_dir"/*.wav; do
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
                               [1:a]aformat=sample_fmts=fltp:sample_rates=32000:channel_layouts=mono,aloop=-1:2147483647:0[a2]; \
                               [a1][a2]amix=inputs=2:weights=${perf_vol} ${noise_vol}[aout]" \
                -map "[aout]" \
                -ar 32000 -ac 1 "$output_path" -y
            
            echo "Created: $output_path"
        done
    done
done

echo "Processing complete. Dataset saved in: $output_dir"

# Create metadata file
metadata_file="${output_dir}/metadata.json"
echo "{" > "$metadata_file"
echo "  \"dataset_info\": {" >> "$metadata_file"
echo "    \"name\": \"${dataset_name}\"," >> "$metadata_file"
echo "    \"performance_source\": \"${performance_dir}\"," >> "$metadata_file"
echo "    \"noise_source\": \"${noise_dir}\"," >> "$metadata_file"
echo "    \"mix_ratios\": [${mix_ratios[@]}]" >> "$metadata_file"
echo "  }" >> "$metadata_file"
echo "}" >> "$metadata_file"

echo "Metadata saved to: $metadata_file"
