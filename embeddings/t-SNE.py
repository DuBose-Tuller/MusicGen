import numpy as np
from sklearn.manifold import TSNE
import os
import yaml
import argparse
from datetime import datetime
import hashlib
import json
from h5py import File as H5File
from h5_processor import H5DataProcessor, DatasetConfig

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="T-SNE dimensionality reduction of embeddings")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to process')
    parser.add_argument('--perplexity', type=float, help='T-SNE perplexity parameter')
    parser.add_argument('--n_components', type=int, default=2, help='Number of dimensions to reduce to')
    parser.add_argument('--n_iter', type=int, help='Number of iterations for optimization')
    parser.add_argument('--output', help='Output filename for the results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()

def merge_config(file_config, args):
    # Start with default T-SNE parameters if not in config
    if 'tsne' not in file_config:
        file_config['tsne'] = {
            'perplexity': 30.0,
            'n_components': 5,
            'n_iter': 1000
        }
    
    # Override with command line arguments if provided
    if args.datasets:
        file_config['datasets'] = [d for d in file_config['datasets'] if d['dataset'] in args.datasets]
    if args.perplexity:
        file_config['tsne']['perplexity'] = args.perplexity
    if args.n_components:
        file_config['tsne']['n_components'] = args.n_components
    if args.n_iter:
        file_config['tsne']['n_iter'] = args.n_iter
    
    return file_config

def generate_output_filename(config):
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tsne_{timestamp}_{config_hash}.h5"

def save_results(tsne_embeddings, labels, class_names, config, output_path):
    """Save T-SNE embeddings and metadata to H5 file."""
    with H5File(output_path, 'w') as f:
        # Save embeddings
        f.create_dataset('embeddings', data=tsne_embeddings)
        f.create_dataset('labels', data=labels)
        f.create_dataset('class_names', data=np.array(class_names, dtype='S'))
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        
        # Convert metadata to JSON string and save as attribute
        f.attrs['metadata'] = json.dumps(metadata)

def main():
    # Parse arguments and load config
    args = parse_arguments()
    file_config = load_config(args.config)
    config = merge_config(file_config, args)

    # Create output directory
    output_dir = os.path.join("../results", "tsne")
    os.makedirs(output_dir, exist_ok=True)

    # Process datasets using H5DataProcessor
    processor = H5DataProcessor(verbose=args.verbose)
    datasets = processor.process_configs(config['datasets'])
    X, y, class_names = processor.combine_datasets(datasets)

    if args.verbose:
        print(f"\nPerforming T-SNE with parameters:")
        print(f"  Perplexity: {config['tsne']['perplexity']}")
        print(f"  Components: {config['tsne']['n_components']}")
        print(f"  Iterations: {config['tsne']['n_iter']}")

    # Create and fit T-SNE model
    tsne = TSNE(
        n_components=config['tsne']['n_components'],
        perplexity=config['tsne']['perplexity'],
        n_iter=config['tsne']['n_iter'],
        verbose=args.verbose
    )
    tsne_embeddings = tsne.fit_transform(X)

    # Generate output filename and save results
    output_filename = args.output if args.output else generate_output_filename(config)
    output_path = os.path.join(output_dir, output_filename)
    save_results(tsne_embeddings, y, class_names, config, output_path)

    if args.verbose:
        print(f"\nResults saved to: {output_path}")
        print(f"Embedded {len(y)} samples to {config['tsne']['n_components']} dimensions")
        print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")

if __name__ == "__main__":
    main()
