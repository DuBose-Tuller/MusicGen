# Noise Analysis Results

This directory contains the results of noise impact analysis experiments on audio embeddings. The analysis examines how different types and levels of noise affect the distance between embeddings in a reduced dimensional space.

## File Naming Convention

Files are named using the following pattern:

```
noise_impact_{reference_name}_{timestamp}_distance.png
```

- `reference_name`: Name of the reference audio file, with original extension removed
- `timestamp`: Format `YYYYMMDD_HHMMSS` indicating when the analysis was run

For example:
```
noise_impact_Bach_45_CPM_20240312_143021_distance.png
```
means:
- Reference file: "Bach_45_CPM.wav"
- Generated on March 12, 2024 at 14:30:21

## File Types

For each analysis run:

### PNG Files
- `*_distance.png`: Plot showing embedding distances vs SNR for different noise types
  - X-axis: Signal-to-Noise Ratio (SNR) in dB
  - Y-axis: Euclidean distance in reduced embedding space
  - Multiple lines if multiple noise files were used
  - Legend identifying each noise source

### CSV Files
- `*_{noise_name}_data.csv`: Raw numerical data for each reference-noise combination
  - Column 1: SNR (dB)
  - Column 2: Distance from clean embedding
  - Header includes metadata about the specific analysis
  - Length of audio used (in seconds)

## Interpretation

- Lower SNR (more negative) means more noise relative to signal
- Higher distances indicate greater deviation from the clean reference embedding
- Each line in a plot represents a different noise type applied to the same reference audio
- The timestamp allows tracking different experimental runs

## Analysis Parameters
- Dimensionality reduction: UMAP to 5D (default)
- SNR range: -20dB to +40dB (default)
- Model: MusicGen-melody
- Uses shorter length when reference and noise files differ in duration
