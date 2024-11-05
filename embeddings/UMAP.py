import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os
import yaml
import argparse
from datetime import datetime
import hashlib
import json
from h5_processor import H5DataProcessor, DatasetConfig

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="UMAP visualization of embeddings")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to process')
    parser.add_argument('--n_neighbors', type=int, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, help='UMAP min_dist parameter')
    parser.add_argument('--output', help='Output filename for the visualization')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()

def merge_config(file_config, args):
    if args.datasets:
        file_config['datasets'] = [d for d in file_config['datasets'] if d['dataset'] in args.datasets]
    if args.n_neighbors:
        file_config['umap']['n_neighbors'] = args.n_neighbors
    if args.min_dist:
        file_config['umap']['min_dist'] = args.min_dist
    return file_config

def generate_output_filename(config):
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"UMAP_{timestamp}_{config_hash}.png"

def save_metadata(config, output_path):
    """Save metadata in YAML format.
    
    Args:
        config: Configuration dictionary
        output_path: Path to the output image file
    """
    def convert_numpy(obj):
        """Convert numpy types to python native types for YAML serialization."""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config
    }
    
    # Convert all numpy types to native Python types
    metadata = {k: convert_numpy(v) for k, v in metadata.items()}
    
    metadata_filename = os.path.splitext(output_path)[0] + "_metadata.yaml"
    
    with open(metadata_filename, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

def main():
    args = parse_arguments()
    file_config = load_config(args.config)
    config = merge_config(file_config, args)

    # Create output directory
    output_dir = os.path.join("../results", "UMAP")
    os.makedirs(output_dir, exist_ok=True)

    # Process datasets using H5DataProcessor
    processor = H5DataProcessor(verbose=args.verbose)
    datasets = processor.process_configs(config['datasets'])
    X, y, class_names = processor.combine_datasets(datasets)

    # Generate output filename and path
    output_filename = generate_output_filename(config)
    output_path = os.path.join(output_dir, output_filename)

    # Create and fit UMAP model
    umap_model = UMAP(
        n_neighbors=config['umap']['n_neighbors'],
        min_dist=config['umap']['min_dist']
    )
    umap_embeddings = umap_model.fit_transform(X)

    # Create visualization
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab20')
    
    for i, class_name in enumerate(class_names):
        mask = y == i
        plt.scatter(
            umap_embeddings[mask, 0],
            umap_embeddings[mask, 1],
            c=[cmap(i/len(class_names))],
            label=class_name,
            alpha=0.7
        )

    plt.legend(title="Datasets")
    plt.title("UMAP Visualization of Embeddings")

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    # Save metadata
    save_metadata(config, output_path)

    print(f"UMAP visualization has been saved as '{output_path}'")
    print(f"Metadata has been saved as '{os.path.splitext(output_path)[0]}_metadata.json'")

if __name__ == "__main__":
    main()