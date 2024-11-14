import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchaudio
import argparse
from pathlib import Path
from datetime import datetime
import json
from audiocraft_fork.audiocraft.models import MusicGen
from embeddings.state_manager import state_manager
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from umap import UMAP

def load_audio(file_path, device="cuda"):
    """Load audio file and prepare for model."""
    waveform, sr = torchaudio.load(file_path)
    if sr != 32000:
        waveform = torchaudio.transforms.Resample(sr, 32000)(waveform)
    
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze noise impact on audio embeddings")
    parser.add_argument('--mixing-dir', required=True, help='Path to directory containing mixed audio files')
    parser.add_argument('--reduced-dim', type=int, default=5, help='Dimensionality after reduction')
    parser.add_argument('--output', default='../results/noise_analysis', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()

def load_mixing_metadata(mixing_dir):
    """Load metadata from mixing run directory."""
    config_path = Path(mixing_dir) / "config.json"
    if not config_path.exists():
        raise ValueError(f"No config.json found in {mixing_dir}")
    
    with open(config_path) as f:
        return json.load(f)

def group_mixed_files(mixing_dir):
    """Group mixed files by reference file and noise file."""
    grouped_files = {}
    
    for file in Path(mixing_dir).glob("*.wav"):
        # Skip if no corresponding metadata file
        metadata_file = file.with_suffix('.wav.json')
        if not metadata_file.exists():
            continue
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        ref_file = metadata['reference_file']
        noise_file = metadata['noise_file']
        snr = metadata['snr_db']
        
        key = (ref_file, noise_file)
        if key not in grouped_files:
            grouped_files[key] = []
        
        grouped_files[key].append((file, snr))
    
    # Sort each group by SNR
    for key in grouped_files:
        grouped_files[key].sort(key=lambda x: x[1])
    
    return grouped_files

def process_file_group(ref_noise_pair, file_snr_pairs, model, reduced_dim, verbose=False):
    """Process a group of files with the same reference and noise."""
    ref_file, noise_file = ref_noise_pair
    if verbose:
        print(f"\nProcessing group:\nReference: {ref_file}\nNoise: {noise_file}")
    
    # Get reference embedding from the original file
    ref_path = Path(ref_file)
    ref_wave = load_audio(ref_path)
    ref_embedding = get_embedding(ref_wave, model)
    
    # Get embeddings for all noisy versions
    embeddings = [ref_embedding]  # Start with reference embedding
    snr_levels = []
    
    for file_path, snr in file_snr_pairs:
        if verbose:
            print(f"Processing {file_path.name} (SNR: {snr}dB)")
        
        wave = load_audio(file_path)
        emb = get_embedding(wave, model)
        embeddings.append(emb)
        snr_levels.append(snr)
    
    # Convert to numpy array and reduce dimensionality
    embeddings = np.array(embeddings)
    if verbose:
        print(f"Reducing dimensionality from {embeddings.shape[1]} to {reduced_dim}")
    
    # Initialize and fit UMAP
    umap_model = UMAP(
        n_components=reduced_dim,
        metric='euclidean',
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    # Calculate distances in reduced space
    reference_embedding = reduced_embeddings[0]  # First embedding is the reference
    distances = [euclidean(reference_embedding, emb) for emb in reduced_embeddings[1:]]
    
    return {
        'reference': ref_file,
        'noise': noise_file,
        'snr_levels': snr_levels,
        'distances': distances,
        'length_seconds': wave.shape[-1] / 32000  # Use last processed wave for length
    }

def plot_results(results, output_dir):
    """Plot and save the results, grouped by reference file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Group results by reference file
    grouped_results = {}
    for result in results:
        ref_path = result['reference']
        if ref_path not in grouped_results:
            grouped_results[ref_path] = []
        grouped_results[ref_path].append(result)
    
    # Create one plot per reference file
    for ref_path, ref_results in grouped_results.items():
        ref_name = Path(ref_path).stem
        output_name = f"noise_impact_{ref_name}_{timestamp}"
        
        plt.figure(figsize=(12, 7))
        
        # Plot line for each noise file
        for result in ref_results:
            noise_name = Path(result['noise']).stem
            plt.plot(result['snr_levels'], result['distances'], 
                    marker='o', label=noise_name)
        
        plt.xlabel('Signal-to-Noise Ratio (dB)')
        plt.ylabel('Distance from Clean Embedding (reduced space)')
        plt.title(f'Embedding Distance vs. Noise Level\n{ref_name}\n'
                 f'Length: {ref_results[0]["length_seconds"]:.2f}s')
        plt.grid(True)
        plt.legend(title='Noise Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{output_name}_distance.png"), 
                   bbox_inches='tight')
        plt.close()
        
        # Save numerical results
        for result in ref_results:
            noise_name = Path(result['noise']).stem
            results_data = np.column_stack((result['snr_levels'], result['distances']))
            header = (f"# Reference: {result['reference']}\n"
                     f"# Noise: {result['noise']}\n"
                     f"# Length: {result['length_seconds']:.2f}s\n"
                     f"SNR_dB,Embedding_Distance")
            np.savetxt(os.path.join(output_dir, f"{output_name}_{noise_name}_data.csv"),
                      results_data,
                      delimiter=",",
                      header=header,
                      comments="")

def main():
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mixing configuration and metadata
    mixing_metadata = load_mixing_metadata(args.mixing_dir)
    grouped_files = group_mixed_files(args.mixing_dir)
    
    if args.verbose:
        print(f"Found {len(grouped_files)} reference-noise pairs")
    
    # Load model
    print("Loading MusicGen model...")
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    
    # Process all groups
    results = []
    for ref_noise_pair, file_snr_pairs in grouped_files.items():
        result = process_file_group(
            ref_noise_pair, file_snr_pairs, model, args.reduced_dim, args.verbose
        )
        results.append(result)
    
    # Plot and save results
    plot_results(results, output_dir)
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "mixing_run": mixing_metadata,
        "reduced_dim": args.reduced_dim
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()