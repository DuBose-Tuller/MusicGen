import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def load_umap_data(h5_path):
    """Load UMAP embeddings and metadata from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        embeddings = f['embeddings'][:]
        labels = f['labels'][:]
        # Decode bytes to strings for class names
        class_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                      for name in f['class_names'][:]]
    return embeddings, labels, class_names

def calculate_distances(embeddings):
    """Calculate pairwise Euclidean distances between all embeddings."""
    return squareform(pdist(embeddings))

def plot_distance_matrix(distances, output_path, title="Pairwise Euclidean Distances"):
    """Create and save a heatmap of the distance matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(distances, cmap='viridis', 
                xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    
    # Add colorbar label
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Euclidean Distance')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_average_distances(distances, labels, class_names, output_path):
    """Create and save a heatmap of average distances between classes."""
    n_classes = len(class_names)
    class_avg_distances = np.zeros((n_classes, n_classes))
    
    # Calculate average distances between classes
    for i in range(n_classes):
        for j in range(n_classes):
            mask_i = labels == i
            mask_j = labels == j
            class_distances = distances[mask_i][:, mask_j]
            class_avg_distances[i, j] = np.mean(class_distances)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(class_avg_distances, annot=True, fmt='.3f',
                xticklabels=class_names, yticklabels=class_names,
                cmap='viridis')
    plt.title('Average Distances Between Classes')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze pairwise distances in UMAP embeddings")
    parser.add_argument('input', help='Path to H5 file containing UMAP embeddings')
    parser.add_argument('--output-dir', default='pairwise_analysis',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}")
    embeddings, labels, class_names = load_umap_data(args.input)
    print(f"Loaded {len(embeddings)} embeddings with {len(class_names)} classes")
    
    # Calculate distances
    print("Calculating pairwise distances...")
    distances = calculate_distances(embeddings)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_distance_matrix(
        distances,
        output_dir / "full_distance_matrix.png",
        f"Pairwise Distances ({len(embeddings)} samples)"
    )
    
    plot_class_average_distances(
        distances,
        labels,
        class_names,
        output_dir / "class_average_distances.png"
    )
    
    # Save numerical results
    print("Saving numerical results...")
    np.save(output_dir / "distances.npy", distances)
    
    # Calculate and save summary statistics
    stats = {
        "global_stats": {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances[distances > 0])),  # Exclude self-distances
            "max": float(np.max(distances))
        },
        "class_stats": {}
    }
    
    # Calculate per-class statistics
    for i, class_name in enumerate(class_names):
        mask = labels == i
        class_distances = distances[mask][:, mask]
        stats["class_stats"][class_name] = {
            "mean_internal": float(np.mean(class_distances[class_distances > 0])),
            "std_internal": float(np.std(class_distances[class_distances > 0])),
            "sample_count": int(np.sum(mask))
        }
    
    # Save stats to file
    np.save(output_dir / "distance_stats.npy", stats)
    
    print(f"Results saved to {output_dir}")
    print("\nSummary statistics:")
    print(f"Global mean distance: {stats['global_stats']['mean']:.3f}")
    print(f"Global std deviation: {stats['global_stats']['std']:.3f}")
    print("\nPer-class sample counts:")
    for class_name, class_stat in stats["class_stats"].items():
        print(f"{class_name}: {class_stat['sample_count']} samples")

if __name__ == "__main__":
    main()