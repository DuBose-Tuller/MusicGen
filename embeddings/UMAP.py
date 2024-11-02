import h5py
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os
import yaml
import argparse
from datetime import datetime
import hashlib

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
    return parser.parse_args()

def merge_config(file_config, args):
    if args.datasets:
        file_config['datasets'] = [d for d in file_config['datasets'] if d['dataset'] in args.datasets]
    if args.n_neighbors:
        file_config['umap']['n_neighbors'] = args.n_neighbors
    if args.min_dist:
        file_config['umap']['min_dist'] = args.min_dist
    if args.output:
        file_config['output']['filename'] = args.output
    return file_config

def load_embeddings(filename, dataset_config):
    """Load embeddings from H5 file, handling potential subfolder structure."""
    embeddings = []
    labels = []
    dataset_name = dataset_config['dataset']
    merge_subfolders = dataset_config.get('merge_subfolders', True)  # Default to merging
    
    with h5py.File(filename, 'r') as f:
        embeddings_group = f['embeddings']
        
        if len(embeddings_group.keys()) == 0:
            return np.array([]), []
            
        # Check if we have a hierarchical structure
        has_subfolders = any(isinstance(embeddings_group[key], h5py.Group) 
                           for key in embeddings_group.keys())
        
        if has_subfolders and not merge_subfolders:
            # Handle hierarchical structure with separate classes for subfolders
            for group_name in embeddings_group.keys():
                group = embeddings_group[group_name]
                if isinstance(group, h5py.Group):
                    for name in group.keys():
                        embeddings.append(group[name][()])
                        labels.append(f"{dataset_name}/{group_name}")
                else:
                    # Handle any files in root level
                    embeddings.append(group[()])
                    labels.append(dataset_name)
        else:
            # Either no subfolders or merging subfolders
            def process_group(group, current_path=""):
                for name in group.keys():
                    item = group[name]
                    if isinstance(item, h5py.Group):
                        process_group(item, current_path)
                    else:
                        embeddings.append(item[()])
                        labels.append(dataset_name)
                        
            process_group(embeddings_group)
    
    return np.array(embeddings), labels

def get_dataset_name(filename, config_entry):
    """Get dataset name, considering potential subfolder structure."""
    base_name = os.path.basename(os.path.dirname(os.path.dirname(filename)))
    merge_subfolders = config_entry.get('merge_subfolders', True)
    return base_name if merge_subfolders else f"{base_name}/*"

def get_filenames(config):
    files = []
    for data in config['datasets']:
        if data['segment'] and data['stride']:
            sampling = f"s{data['segment']}-t{data['stride']}"
        else:
            sampling = "raw"
        path = os.path.join(data['dataset'], sampling)
        if os.path.exists(path):
            filename = os.path.join(path, f"{data['method']}_embeddings.h5")
            files.append(filename)       

    if not files:
        raise ValueError("No valid files found")

    return files

def generate_output_filename(config):
    # Create a unique identifier based on the configuration
    config_str = yaml.dump(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the filename
    filename = f"UMAP_{timestamp}_{config_hash}.png"
    
    return filename

def save_metadata(config, output_path):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config
    }
    
    metadata_filename = os.path.splitext(output_path)[0] + "_metadata.yaml"
    
    with open(metadata_filename, 'w') as f:
        yaml.dump(metadata, f, indent=2)

def main():
    args = parse_arguments()
    file_config = load_config(args.config)
    config = merge_config(file_config, args)

    # Create output directory
    output_dir = os.path.join("../results", "UMAP")
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    output_filename = generate_output_filename(config)
    output_path = os.path.join(output_dir, output_filename)

    json_files = get_filenames(config)
    print("Processing files:", json_files)

    all_embeddings = []
    all_labels = []
    dataset_names = []

    for i, (file, dataset_config) in enumerate(zip(json_files, config['datasets'])):
        embeddings, labels = load_embeddings(file, dataset_config)
        if len(embeddings) > 0:
            all_embeddings.append(embeddings)
            all_labels.extend(labels)
            dataset_names.extend(list(set(labels)))  # Get unique labels

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    umap_model = UMAP(n_neighbors=config['umap']['n_neighbors'], min_dist=config['umap']['min_dist'])
    umap_embeddings = umap_model.fit_transform(all_embeddings)

    plt.figure(figsize=(12, 8))

    # Use a colormap that can handle an arbitrary number of datasets
    cmap = plt.get_cmap('tab20')  # This colormap supports up to 20 distinct colors
    
    # Create a scatter plot for each unique label
    unique_labels = sorted(set(all_labels))
    for i, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        plt.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                   c=[cmap(i/len(unique_labels))], label=label, alpha=0.7)

    plt.legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("UMAP Visualization of Embeddings")

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    # Save metadata
    save_metadata(config, output_path)

    print(f"UMAP visualization has been saved as '{output_path}'")
    print(f"Metadata has been saved as '{os.path.splitext(output_path)[0]}_metadata.yaml'")

if __name__ == "__main__":
    main()