#!/bin/bash

# Function to cleanup test data
cleanup() {
    rm -rf test_data
}

# Function to handle errors
handle_error() {
    echo "Error: $1"
    cleanup
    exit 1
}

test_trim_audio() {
    # Create test directory structure
    mkdir -p test_data/dataset1/raw || handle_error "Failed to create test directory"
    
    # Create a test audio file
    ffmpeg -f lavfi -i "sine=frequency=1000:duration=10" test_data/dataset1/raw/test.wav || \
        handle_error "Failed to create test audio file"
    
    # Test with different segment lengths
    ./trim_audio.sh -s 2 -t 1 test_data/dataset1 || handle_error "Failed to run trim_audio.sh"
    
    # Check if output directory exists
    if [ ! -d "test_data/dataset1/s2-t1" ]; then
        handle_error "Output directory not created"
    fi
    
    # Check if files were created
    file_count=$(ls test_data/dataset1/s2-t1/*.wav 2>/dev/null | wc -l)
    if [ $file_count -eq 0 ]; then
        handle_error "No output files created"
    fi
    
    # Verify file durations
    for file in test_data/dataset1/s2-t1/*.wav; do
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
        if [ $(echo "$duration > 2.1" | bc -l) -eq 1 ] || [ $(echo "$duration < 1.9" | bc -l) -eq 1 ]; then
            handle_error "File $file has incorrect duration: $duration (expected ~2.0)"
        fi
    done
    
    echo "All tests passed!"
    cleanup
}

# Set up trap for cleanup on script exit
trap cleanup EXIT

# Run tests
test_trim_audio