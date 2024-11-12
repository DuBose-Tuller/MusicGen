import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchaudio
import argparse
from pathlib import Path
from datetime import datetime
import yaml
from audiocraft_fork.audiocraft.models import MusicGen
from embeddings.state_manager import state_manager
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from umap import UMAP

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze noise impact on audio embeddings")
    parser.add_argument('--reference', required=True, help='Path to reference audio file or directory')
    parser.add_argument('--noise', required=True, help='Path to noise file or directory')
    parser.add_argument('--min-snr', type=float, default=-20, help='Minimum SNR in dB')
    parser.add_argument('--max-snr', type=float, default=40, help='Maximum SNR in dB')
    parser.add_argument('--snr-steps', type=int, default=13, help='Number of SNR steps')
    parser.add_argument('--reduced-dim', type=int, default=5, help='Dimensionality after reduction')
    parser.add_argument('--output', default='../results/noise_analysis', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()

def load_and_normalize_audio(file_path, target_sr=32000):
    """Load and normalize audio file."""
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Normalize to [-1, 1]
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform, waveform.shape[1]

def prepare_audio_pair(clean_wave, noise_wave, clean_length, noise_length):
    """Prepare audio pair to have matching lengths using the shorter length."""
    # Use the shorter length
    target_length = min(clean_length, noise_length)
    
    # Trim both to target length
    clean_wave = clean_wave[:, :target_length]
    
    # For noise, if it's shorter than target, repeat it
    if noise_length < target_length:
        repeats = target_length // noise_length + 1
        noise_wave = noise_wave.repeat(1, repeats)
    
    # Trim noise to exact length
    noise_wave = noise_wave[:, :target_length]
    
    return clean_wave, noise_wave, target_length

def mix_noise(clean_wave, noise_wave, snr_db):
    """Mix noise with signal at specified SNR level."""
    signal_power = torch.mean(clean_wave ** 2)
    noise_power = torch.mean(noise_wave ** 2)
    
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = torch.sqrt(target_noise_power / noise_power)
    scaled_noise = noise_wave * scaling_factor
    
    noisy_signal = clean_wave + scaled_noise
    noisy_signal = noisy_signal / torch.max(torch.abs(noisy_signal))
    return noisy_signal

def get_embedding(waveform, model, device="cuda"):
    """Get embedding for audio using MusicGen model."""
    state_manager.clear_embedding()
    state_manager.set_method("last")
    
    waveform = waveform.unsqueeze(0).to(device)
    duration = waveform.shape[-1] / 32000 + 0.04
    
    model.set_generation_params(duration=duration)
    model.generate_continuation(waveform, 32000)
    
    embedding = state_manager.get_embedding()
    return embedding.numpy()

def process_file_pair(clean_file, noise_file, snr_range, model, reduced_dim, verbose=False):
    """Process a single pair of reference and noise files."""
    if verbose:
        print(f"\nProcessing:\nReference: {clean_file}\nNoise: {noise_file}")
    
    # Load audio files
    clean_wave, clean_length = load_and_normalize_audio(clean_file)
    noise_wave, noise_length = load_and_normalize_audio(noise_file)
    
    if verbose:
        print(f"Clean audio length: {clean_length/32000:.2f}s")
        print(f"Noise audio length: {noise_length/32000:.2f}s")
        print(f"Using length: {min(clean_length, noise_length)/32000:.2f}s")
    
    # Prepare audio to same length
    clean_wave, noise_wave, used_length = prepare_audio_pair(
        clean_wave, noise_wave, clean_length, noise_length)
    
    # Get embeddings for all SNR levels
    embeddings = []
    clean_embedding = get_embedding(clean_wave, model)
    embeddings.append(clean_embedding)
    
    for snr in snr_range:
        if verbose:
            print(f"Processing SNR: {snr}dB")
        noisy_wave = mix_noise(clean_wave, noise_wave, snr)
        noisy_embedding = get_embedding(noisy_wave, model)
        embeddings.append(noisy_embedding)
    
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
        'reference': str(clean_file),
        'noise': str(noise_file),
        'distances': distances,
        'length_seconds': used_length / 32000
    }

def plot_results(results, snr_range, output_dir):
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
            plt.plot(snr_range, result['distances'], marker='o', label=noise_name)
        
        plt.xlabel('Signal-to-Noise Ratio (dB)')
        plt.ylabel('Distance from Clean Embedding (reduced space)')
        plt.title(f'Embedding Distance vs. Noise Level\n{ref_name}\n'
                 f'Length: {ref_results[0]["length_seconds"]:.2f}s')
        plt.grid(True)
        plt.legend(title='Noise Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{output_name}_distance.png"), bbox_inches='tight')
        plt.close()
        
        # Save numerical results
        for result in ref_results:
            noise_name = Path(result['noise']).stem
            results_data = np.column_stack((snr_range, result['distances']))
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
    os.makedirs(args.output, exist_ok=True)
    
    # Setup experiment parameters
    snr_range = np.linspace(args.min_snr, args.max_snr, args.snr_steps)
    
    # Load model
    print("Loading MusicGen model...")
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    
    # Get file lists
    reference_path = Path(args.reference)
    noise_path = Path(args.noise)
    
    reference_files = [reference_path] if reference_path.is_file() else list(reference_path.glob('*.wav'))
    noise_files = [noise_path] if noise_path.is_file() else list(noise_path.glob('*.wav'))
    
    if not reference_files or not noise_files:
        raise ValueError(f"No WAV files found in specified paths: {reference_files} or {noise_files}")
    
    # Process all combinations
    results = []
    for ref_file in reference_files:
        for noise_file in noise_files:
            result = process_file_pair(
                ref_file, noise_file, snr_range, model, args.reduced_dim, args.verbose
            )
            results.append(result)
    
    # Plot and save results
    plot_results(results, snr_range, args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()