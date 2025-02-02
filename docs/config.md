# Configuration System Documentation

## Overview
The configuration system uses YAML files to specify datasets and analysis parameters for embedding generation, classification, and dimensionality reduction tasks. The system supports both simple dataset organization and hierarchical structures with automatic attribute detection from folder structures.

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
- `subfolder`: (Optional) Name of the subfolder or path containing the audio files. Defaults to "raw"
- `merge_subfolders`: (Optional) Boolean indicating whether to treat all subfolders as one class. Defaults to true

### Organization Methods

#### Traditional (Flat) Structure
```
data/
└── dataset_name/
    ├── raw/
    └── technical_attributes/
```

#### Hierarchical Structure
```
data/
└── dataset_name/
    └── subclass/
        └── technical_attributes/
```

### Subfolder Naming Convention
The system automatically detects attributes from the final subfolder name using these patterns:
- Segment length: `s<number>` (e.g., "s15" for 15-second segments)
- Stride length: `t<number>` (e.g., "t15" for 15-second stride)
- Reversed audio: `reversed`
- Noise level: `noise<number>` (e.g., "noise0.5" for 0.5 noise level)

Parts can be separated by either hyphens or underscores:
```
s15-t15_reversed
s15_t15_noise0.5
```

### Class Name Generation

#### Traditional Structure
- Basic dataset: `"dataset_name"`
- With attributes: `"dataset_name_reversed"` or `"dataset_name_noise0.5"`

#### Hierarchical Structure
The system combines:
1. Dataset name
2. Subclass folder name(s)
3. Technical attributes (if distinguishing)

Examples:
- `NHS/Dance/s15-t15` → `NHS_Dance`
- `NHS/Dance/s15-t15_reversed` → `NHS_Dance_reversed`
- `NHS/Dance/s15-t15_noise0.5` → `NHS_Dance_noise0.5`

### Example Configurations

#### 1. Traditional Structure
```yaml
# Basic dataset
datasets:
  - dataset: "acpas"

# With specific subfolder
datasets:
  - dataset: "acpas"
    subfolder: "s15-t15"
    merge_subfolders: false

# Multiple versions
datasets:
  - dataset: "acpas"
    subfolder: "s15-t15_reversed"
    merge_subfolders: false
  - dataset: "acpas"
    subfolder: "s15-t15"
    merge_subfolders: false
```

#### 2. Hierarchical Structure
```yaml
# Single subclass
datasets:
  - dataset: "NHS"
    subfolder: "Dance/s15-t15"
    merge_subfolders: false

# Multiple subclasses
datasets:
  - dataset: "NHS"
    subfolder: "Dance/s15-t15"
    merge_subfolders: false
  - dataset: "NHS"
    subfolder: "Love/s15-t15"
    merge_subfolders: false

# Mixed attributes
datasets:
  - dataset: "NHS"
    subfolder: "Dance/s15-t15"
    merge_subfolders: false
  - dataset: "NHS"
    subfolder: "Dance/s15-t15_reversed"
    merge_subfolders: false
```

#### 3. Mixed Usage
```yaml
datasets:
  # Hierarchical dataset
  - dataset: "NHS"
    subfolder: "Dance/s15-t15"
    merge_subfolders: false
  
  # Traditional dataset
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

## Directory Structure

### Traditional Structure
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

### Hierarchical Structure
```
data/
└── NHS/
    ├── Dance/
    │   └── s15-t15/
    ├── Love/
    │   └── s15-t15/
    ├── Lullaby/
    │   └── s15-t15/
    └── Healing/
        └── s15-t15/

embeddings/
└── NHS/
    ├── Dance/
    │   └── s15-t15/
    ├── Love/
    │   └── s15-t15/
    ├── Lullaby/
    │   └── s15-t15/
    └── Healing/
        └── s15-t15/
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
  - dataset: "NHS"
    subfolder: "Dance/s15-t15"
    merge_subfolders: false
  - dataset: "NHS"
    subfolder: "Love/s15-t15"
    merge_subfolders: false
  - dataset: "birdsong"
    subfolder: "raw"
    merge_subfolders: true

umap:
  n_neighbors: 15
  min_dist: 0.1
  n_components: 2
```