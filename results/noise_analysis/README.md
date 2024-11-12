# Noise Analysis Results

This directory contains the results of noise impact analysis experiments on audio embeddings. The analysis examines how different types and levels of noise affect the distance between embeddings in a reduced dimensional space.

## Directory Structure

Each experimental run creates a new timestamped directory:
```
results/
└── noise_analysis/
    ├── run_20240312_143021/
    │   ├── noise_impact_reference1_distance.png
    │   ├── noise_impact_reference1_noise1_data.csv
    │   ├── noise_impact_reference1_noise2_data.csv
    │   └── ...
    └── run_20240312_150442/
        └── ...
```

## File Naming Convention

Within each run directory, files are named using the following patterns:

### Plot Files
```
noise_impact_{reference_name}_distance.png
```
- `reference_name`: Name of the reference audio file, with original extension removed
- Each plot shows multiple lines, one for each noise file tested

### Data Files
```
noise_impact_{reference_name}_{noise_name}_data.csv
```
- `reference_name`: Name of the reference audio file
- `noise_name`: Name of the noise file used

## File Contents

### PNG Files
- `*_distance.png`: Plot showing embedding distances vs SNR for different noise types
  - X-axis: Signal-to-Noise Ratio (SNR) in dB (-20 to +40 by default)
  - Y-axis: Euclidean distance in reduced embedding space
  - Multiple lines showing different noise sources
  - Legend identifying each noise source

### CSV Files
- `*_data.csv`: Raw numerical data for each reference-noise combination
  - Column 1: SNR (dB)
  - Column 2: Distance from clean embedding
  - Header includes:
    - Reference file path
    - Noise file path
    - Length of audio used (in seconds)

## Dimensionality Reduction

The analysis uses UMAP for dimensionality reduction with the following characteristics:
- All embeddings from a single run are reduced together in one UMAP transformation
- Number of points used = (# reference files × # noise files × 14)
  - 14 = 1 clean reference + 13 noisy versions at different SNRs
- Default reduction is to 5 dimensions before calculating distances
- UMAP parameters:
  - n_neighbors: 15
  - min_dist: 0.1
  - metric: euclidean
  - random_state: 42 (for reproducibility)

## Analysis Parameters
- Input embedding dimension: 1538 (MusicGen-melody)
- SNR range: -20dB to +40dB (default)
- Number of SNR steps: 13 (default)
- Audio processing:
  - Sample rate: 32kHz
  - Mono (stereo files are averaged)
  - Uses shorter length when reference and noise files differ in duration
  - Audio normalized to [-1, 1] range

## Running the Analysis

Basic usage:
```bash
python noise_analysis.py \
    --reference path/to/reference.wav \
    --noise path/to/noise.wav \
    --reduced-dim 5 \
    --verbose
```

For multiple files:
```bash
python noise_analysis.py \
    --reference path/to/reference/directory \
    --noise path/to/noise/directory
```

Optional parameters:
- `--min-snr`: Minimum SNR in dB (default: -20)
- `--max-snr`: Maximum SNR in dB (default: 40)
- `--snr-steps`: Number of SNR steps (default: 13)
- `--reduced-dim`: Output dimensionality for UMAP (default: 5)