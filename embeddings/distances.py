import torch
import torchaudio
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from audiocraft.models import MusicGen
from embeddings.state_manager import state_manager
from scipy.spatial.distance import pdist, squareform
from umap import UMAP
from datetime import datetime

def load_audio(file_path, device="cuda"):
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(file_path)
    if sr != 32000:
        waveform = torchaudio.transforms.Resample(sr, 32000)(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform.to(device)

def get_embedding(waveform, model, device="cuda"):
    """Get embedding for audio using MusicGen model."""
    state_manager.clear_embedding()
    state_manager.set_method("last")
    
    waveform = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform
    duration = waveform.shape[-1] / 32000 + 0.04
    
    model.set_generation_params(duration=duration)
    model.generate_continuation(waveform, 32000)
    
    embedding = state_manager.get_embedding()
    return embedding.numpy()

def reduce_dimensionality(embeddings, n_components=5, n_runs=10):
    """Perform multiple UMAP reductions and average the distances."""
    distances_sum = None
    
    for i in range(n_runs):
        umap_model = UMAP(
            n_components=n_components,
            metric='euclidean',
            n_neighbors=15,
            min_dist=0.1,
            random_state=i  # Different seed for each run
        )
        reduced = umap_model.fit_transform(embeddings)
        
        # Calculate pairwise distances for this reduction
        distances = squareform(pdist(reduced))
        
        if distances_sum is None:
            distances_sum = distances
        else:
            distances_sum += distances
    
    # Return average distances
    return distances_sum / n_runs

def plot_distance_matrix(distances, labels, output_path, std_devs=None):
    """Create and save a heatmap of the distance matrix."""
    plt.figure(figsize=(12, 10))
    
    # Create annotation text with optional standard deviations
    if std_devs is not None:
        annot = np.array([[f'{d:.3f}\n±{s:.3f}' for d, s in zip(row, std_row)]
                         for row, std_row in zip(distances, std_devs)])
    else:
        annot = np.around(distances, decimals=3)
    
    sns.heatmap(distances, annot=annot, fmt='s', 
                xticklabels=labels, yticklabels=labels,
                cmap='viridis')
    
    plt.title('Average Pairwise Distances in Reduced Space\n'
             f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Calculate average pairwise distances between audio embeddings")
    parser.add_argument('files', nargs='+', help='Audio files to analyze')
    parser.add_argument('--output', default='average_distances.png', help='Output image path')
    parser.add_argument('--reduced-dim', type=int, default=5, help='UMAP output dimensionality')
    parser.add_argument('--umap-runs', type=int, default=10, help='Number of UMAP runs to average')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Load model
    if args.verbose:
        print("Loading MusicGen model...")
    model = MusicGen.get_pretrained('facebook/musicgen-melody')

    # Process each file
    embeddings = []
    labels = []
    
    if args.verbose:
        print("\nProcessing files:")
    
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        if args.verbose:
            print(f"Processing {path.name}...")
        
        try:
            wave = load_audio(file_path)
            emb = get_embedding(wave, model)
            embeddings.append(emb)
            labels.append(path.stem)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not embeddings:
        print("No valid embeddings generated. Exiting.")
        return

    if args.verbose:
        print(f"\nPerforming {args.umap_runs} UMAP reductions...")

    # Calculate average pairwise distances after dimensionality reduction
    embeddings = np.array(embeddings)
    avg_distances = reduce_dimensionality(
        embeddings, 
        n_components=args.reduced_dim,
        n_runs=args.umap_runs
    )

    # Create and save visualization
    plot_distance_matrix(avg_distances, labels, args.output)
    
    if args.verbose:
        print(f"\nResults saved to {args.output}")
        print("\nAverage distances:")
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                print(f"{label1} - {label2}: {avg_distances[i,j]:.3f}")

if __name__ == "__main__":
    main()