# Configuration System Documentation

## Overview
The configuration system uses YAML files to specify datasets and analysis parameters for embedding generation, classification, and dimensionality reduction tasks. The system supports flexible dataset organization with automatic attribute detection from folder structures.

## Configuration File Structure

```yaml
# Common parameters across all analyses
common:
  random_seed: 42

# Dataset definitions
datasets:
  - dataset: "dataset_name"
    subfolder: "subfolder_name"
    merge_subfolders: false

# Analysis-specific parameters
umap:
  n_neighbors: 15
  min_dist: 0.1
  n_components: 2
```

## Dataset Configuration

### Basic Parameters

- `dataset`: (Required) Name of the dataset directory
- `subfolder`: (Optional) Name of the subfolder containing the audio files. Defaults to "raw"
- `merge_subfolders`: (Optional) Boolean indicating whether to treat all subfolders as one class. Defaults to true

### Subfolder Naming Convention

The system automatically detects attributes from subfolder names using the following patterns:

- Segment length: `s<number>` (e.g., "s15" for 15-second segments)
- Stride length: `t<number>` (e.g., "t15" for 15-second stride)
- Reversed audio: `reversed`
- Noise level: `noise<number>` (e.g., "noise0.5" for 0.5 noise level)

Subfolder parts can be separated by either hyphens or underscores:
```
s15-t15_reversed
s15_t15_noise0.5
```

### Example Configurations

1. Basic dataset with default settings:
```yaml
datasets:
  - dataset: "acpas"
```

2. Dataset with specific subfolder:
```yaml
datasets:
  - dataset: "acpas"
    subfolder: "s15-t15"
    merge_subfolders: false
```

3. Multiple versions of the same dataset:
```yaml
datasets:
  - dataset: "acpas"
    subfolder: "s15-t15_reversed"
    merge_subfolders: false
  
  - dataset: "acpas"
    subfolder: "s15-t15"
    merge_subfolders: false
```

## Analysis Parameters

### UMAP Configuration
```yaml
umap:
  n_neighbors: 15
  min_dist: 0.1
  n_components: 2
  metric: "euclidean"
  random_seed: 42
```

### t-SNE Configuration
```yaml
tsne:
  perplexity: 30.0
  early_exaggeration: 12.0
  metric: "euclidean"
```

### Classifier Configuration
```yaml
classifier:
  test_size: 0.2
  class_weight: "balanced"
  max_iter: 1000
  solver: "lbfgs"
```

## Class Name Generation

The system automatically generates class names based on the dataset name and distinguishing features found in the subfolder name:

- Basic dataset: `"dataset_name"`
- Reversed dataset: `"dataset_name_reversed"`
- Dataset with noise: `"dataset_name_noise0.5"`

When `merge_subfolders` is true, all data from the dataset will use the basic dataset name regardless of subfolder attributes.

## Directory Structure

Expected directory structure:
```
data/
├── dataset1/
│   ├── raw/
│   ├── s15-t15/
│   └── s15-t15_reversed/
└── dataset2/
    ├── raw/
    └── s30-t15/

embeddings/
├── dataset1/
│   ├── raw/
│   ├── s15-t15/
│   └── s15-t15_reversed/
└── dataset2/
    ├── raw/
    └── s30-t15/
```

## Usage Examples

### Basic Analysis
```yaml
datasets:
  - dataset: "acpas"
    subfolder: "s15-t15"
    merge_subfolders: false

umap:
  n_neighbors: 50
  min_dist: 1
  n_components: 2
```

### Comparing Forward and Reversed Audio
```yaml
datasets:
  - dataset: "acpas"
    subfolder: "s15-t15_reversed"
    merge_subfolders: false
  
  - dataset: "acpas"
    subfolder: "s15-t15"
    merge_subfolders: false

classifier:
  test_size: 0.2
  class_weight: "balanced"
```

### Multi-Dataset Analysis
```yaml
datasets:
  - dataset: "acpas"
    subfolder: "s15-t15"
    merge_subfolders: false
  
  - dataset: "birdsong"
    subfolder: "raw"
    merge_subfolders: true
  
  - dataset: "CBF"
    subfolder: "s15_t15"
    merge_subfolders: false

umap:
  n_neighbors: 15
  min_dist: 0.1
  n_components: 2
```