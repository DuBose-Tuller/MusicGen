import torch
import torchaudio
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from audiocraft.models import MusicGen
from state_manager import state_manager
from scipy.spatial.distance import pdist, squareform

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

def plot_distance_matrix(distances, labels, output_path):
    """Create and save a heatmap of the distance matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, annot=True, fmt='.3f', 
                xticklabels=labels, yticklabels=labels,
                cmap='viridis')
    plt.title('Pairwise Euclidean Distances between Audio Embeddings')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Calculate pairwise distances between audio embeddings")
    parser.add_argument('files', nargs='+', help='Audio files to analyze')
    parser.add_argument('--output', default='distance_matrix.png', help='Output image path')
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

    # Calculate pairwise distances
    embeddings = np.array(embeddings)
    distances = squareform(pdist(embeddings))

    # Create and save visualization
    plot_distance_matrix(distances, labels, args.output)
    
    if args.verbose:
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
