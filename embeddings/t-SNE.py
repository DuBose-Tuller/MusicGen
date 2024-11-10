import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    parser.add_argument('--n_components', type=int, default=2, help='Final number of dimensions to reduce to')
    parser.add_argument('--n_iter', type=int, help='Number of iterations for optimization')
    parser.add_argument('--pca_components', type=int, help='Number of PCA components for preprocessing')
    parser.add_argument('--random_seed', '-r', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', help='Output filename for the results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()

def merge_config(file_config, args):
    # Start with default parameters if not in config
    if 'tsne' not in file_config:
        file_config['tsne'] = {
            'perplexity': 30.0,
            'n_components': 5,
            'n_iter': 1000,
            'pca_components': 50,  # Default PCA preprocessing dimension
            'random_seed': 42      # Default random seed
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
    if args.pca_components:
        file_config['tsne']['pca_components'] = args.pca_components
    if args.random_seed:
        file_config['tsne']['random_seed'] = args.random_seed
    
    return file_config

def generate_output_filename(config):
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tsne_{timestamp}_{config_hash}.h5"

def preprocess_with_pca(X, n_components, random_seed, verbose=False):
    """Preprocess data with PCA and standardization."""
    if verbose:
        print(f"\nPreprocessing data:")
        print(f"  Initial dimension: {X.shape[1]}")
    
    # First standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Then apply PCA
    pca = PCA(n_components=n_components, random_state=random_seed)
    X_pca = pca.fit_transform(X_scaled)
    
    if verbose:
        print(f"  PCA reduced dimension: {X_pca.shape[1]}")
        print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, pca.explained_variance_ratio_

def save_results(tsne_embeddings, labels, class_names, config, output_path, pca_variance_ratio=None):
    """Save T-SNE embeddings and metadata to H5 file."""
    with H5File(output_path, 'w') as f:
        # Save embeddings and labels
        f.create_dataset('embeddings', data=tsne_embeddings)
        f.create_dataset('labels', data=labels)
        f.create_dataset('class_names', data=np.array(class_names, dtype='S'))
        
        if pca_variance_ratio is not None:
            f.create_dataset('pca_variance_ratio', data=pca_variance_ratio)
        
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

    # Set random seed for reproducibility
    random_seed = config['tsne']['random_seed']
    np.random.seed(random_seed)

    # Process datasets using H5DataProcessor
    processor = H5DataProcessor(verbose=args.verbose)
    datasets = processor.process_configs(config['datasets'])
    X, y, class_names = processor.combine_datasets(datasets)

    if args.verbose:
        print(f"\nParameters:")
        print(f"  Random seed: {random_seed}")
        print(f"  PCA components: {config['tsne']['pca_components']}")
        print(f"  T-SNE components: {config['tsne']['n_components']}")
        print(f"  Perplexity: {config['tsne']['perplexity']}")
        print(f"  Iterations: {config['tsne']['n_iter']}")

    # Preprocess with PCA
    X_pca, variance_ratio = preprocess_with_pca(
        X, 
        n_components=config['tsne']['pca_components'],
        random_seed=random_seed,
        verbose=args.verbose
    )

    # Create and fit T-SNE model
    tsne = TSNE(
        n_components=config['tsne']['n_components'],
        perplexity=config['tsne']['perplexity'],
        n_iter=config['tsne']['n_iter'],
        random_state=random_seed,
        verbose=args.verbose
    )
    tsne_embeddings = tsne.fit_transform(X_pca)

    # Generate output filename and save results
    output_filename = args.output if args.output else generate_output_filename(config)
    output_path = os.path.join(output_dir, output_filename)
    save_results(tsne_embeddings, y, class_names, config, output_path, variance_ratio)

    if args.verbose:
        print(f"\nResults saved to: {output_path}")
        print(f"Embedded {len(y)} samples to {config['tsne']['n_components']} dimensions")
        print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")

if __name__ == "__main__":
    main()