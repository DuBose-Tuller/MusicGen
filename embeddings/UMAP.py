import numpy as np
from umap import UMAP
import os
import yaml
import argparse
from datetime import datetime
import hashlib
import json
from h5py import File as H5File
import matplotlib.pyplot as plt
from h5_processor import H5DataProcessor, DatasetConfig

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="UMAP dimensionality reduction of embeddings")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to process')
    parser.add_argument('--n_neighbors', type=int, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, help='UMAP min_dist parameter')
    parser.add_argument('--n_components', type=int, default=2, help='Number of dimensions to reduce to')
    parser.add_argument('--metric', type=str, default='euclidean', help='Distance metric to use')
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--output', help='Output filename for the results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()

def merge_config(file_config, args):
    if 'umap' not in file_config:
        file_config['umap'] = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 2,
            'metric': 'euclidean',
            'random_seed': None
        }
    
    if args.datasets:
        file_config['datasets'] = [d for d in file_config['datasets'] if d['dataset'] in args.datasets]
    if args.n_neighbors:
        file_config['umap']['n_neighbors'] = args.n_neighbors
    if args.min_dist:
        file_config['umap']['min_dist'] = args.min_dist
    if args.n_components:
        file_config['umap']['n_components'] = args.n_components
    if args.metric:
        file_config['umap']['metric'] = args.metric
    if args.random_seed:
        file_config['umap']['random_seed'] = args.random_seed
    
    return file_config

def generate_output_filename(config, suffix):
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"umap_{timestamp}_{config_hash}{suffix}"

def create_visualization(embeddings, labels, class_names, output_path):
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab20')
    
    # Create random permutation of all data points
    random_indices = np.random.permutation(len(labels))
    
    # Use a single scatter plot with randomized indices
    scatter = plt.scatter(
        embeddings[random_indices, 0],
        embeddings[random_indices, 1],
        c=labels[random_indices],  # Use labels directly
        cmap=cmap,                 # Apply colormap to these values
        alpha=0.3,
        vmin=0,                   # Ensure proper color scaling
        vmax=len(class_names)-1
    )
    
    # Add legend manually
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
              markerfacecolor=cmap(i/(len(class_names)-1)), markersize=10) 
              for i in range(len(class_names))]
   
    plt.xticks([])
    plt.yticks([])

    # class_names = [name.split("/")[0] for name in class_names]

    plt.legend(handles, class_names, title="Datasets")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def save_results(umap_embeddings, labels, class_names, config, output_dir):
    """Save UMAP embeddings and optionally create visualization."""
    # Save embeddings to H5 file
    h5_filename = generate_output_filename(config, '.h5')
    h5_path = os.path.join(output_dir, h5_filename)
    
    with H5File(h5_path, 'w') as f:
        # Save embeddings and labels
        f.create_dataset('embeddings', data=umap_embeddings)
        f.create_dataset('labels', data=labels)
        f.create_dataset('class_names', data=np.array(class_names, dtype='S'))
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        f.attrs['metadata'] = json.dumps(metadata)

    # Create visualization if dimensionality is 2
    if umap_embeddings.shape[1] == 2:
        png_filename = generate_output_filename(config, '.png')
        png_path = os.path.join(output_dir, png_filename)
        create_visualization(umap_embeddings, labels, class_names, png_path)
        return h5_path, png_path
    
    return h5_path, None

def main():
    args = parse_arguments()
    file_config = load_config(args.config)
    config = merge_config(file_config, args)

    # Create output directory
    output_dir = os.path.join("../results", "UMAP")
    os.makedirs(output_dir, exist_ok=True)

    random_seed = config['umap']['random_seed']
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Process datasets using H5DataProcessor
    processor = H5DataProcessor(verbose=args.verbose)
    datasets = processor.process_configs(config['datasets'])
    X, y, class_names = processor.combine_datasets(datasets)

    if args.verbose:
        print(f"\nParameters:")
        print(f"  Random seed: {'Not set' if random_seed is None else random_seed}")
        print(f"  n_neighbors: {config['umap']['n_neighbors']}")
        print(f"  min_dist: {config['umap']['min_dist']}")
        print(f"  n_components: {config['umap']['n_components']}")
        print(f"  metric: {config['umap']['metric']}")
        print(f"  Input dimension: {X.shape[1]}")

    # Create and fit UMAP model
    umap_kwargs = {
        'n_neighbors': config['umap']['n_neighbors'],
        'min_dist': config['umap']['min_dist'],
        'n_components': config['umap']['n_components'],
        'metric': config['umap']['metric'],
        'verbose': args.verbose
    }
    # Only add random_state if seed was specified
    if random_seed is not None:
        umap_kwargs['random_state'] = random_seed
    
    umap_model = UMAP(**umap_kwargs)
    umap_embeddings = umap_model.fit_transform(X)

    # Save results and possibly create visualization
    h5_path, png_path = save_results(umap_embeddings, y, class_names, config, output_dir)

    if args.verbose:
        print(f"\nResults saved to: {h5_path}")
        print(f"Reduced {len(y)} samples from {X.shape[1]} to {config['umap']['n_components']} dimensions")
        if png_path:
            print(f"Visualization saved to: {png_path}")

if __name__ == "__main__":
    main()