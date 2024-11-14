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

def create_run_directory(base_dir):
    """Create a timestamped directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

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
    target_length = min(clean_length, noise_length)
    clean_wave = clean_wave[:, :target_length]
    
    if noise_length < target_length:
        repeats = target_length // noise_length + 1
        noise_wave = noise_wave.repeat(1, repeats)
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

def collect_embeddings(ref_file, noise_file, snr_range, model, verbose=False):
    """Collect embeddings for a reference-noise pair."""
    if verbose:
        print(f"\nProcessing:\nReference: {ref_file}\nNoise: {noise_file}")
    
    # Load audio files
    clean_wave, clean_length = load_and_normalize_audio(ref_file)
    noise_wave, noise_length = load_and_normalize_audio(noise_file)
    
    if verbose:
        print(f"Clean audio length: {clean_length/32000:.2f}s")
        print(f"Noise audio length: {noise_length/32000:.2f}s")
    
    # Prepare audio to same length
    clean_wave, noise_wave, used_length = prepare_audio_pair(
        clean_wave, noise_wave, clean_length, noise_length)
    
    if verbose:
        print(f"Using length: {used_length/32000:.2f}s")
    
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
    
    return {
        'reference': str(ref_file),
        'noise': str(noise_file),
        'embeddings': np.array(embeddings),
        'length_seconds': used_length / 32000
    }

def reduce_embeddings(all_results, reduced_dim, verbose=False):
    """Perform UMAP reduction on all embeddings together."""
    # Collect all embeddings
    all_embeddings = []
    for result in all_results:
        all_embeddings.extend(result['embeddings'])
    all_embeddings = np.array(all_embeddings)
    
    if verbose:
        print(f"\nPerforming UMAP reduction on {len(all_embeddings)} points")
        print(f"Reducing from {all_embeddings.shape[1]} to {reduced_dim} dimensions")
    
    # Initialize and fit UMAP
    umap_model = UMAP(
        n_components=reduced_dim,
        metric='euclidean',
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    reduced = umap_model.fit_transform(all_embeddings)
    
    # Distribute reduced embeddings back to results
    idx = 0
    for result in all_results:
        n_embeddings = len(result['embeddings'])
        result['reduced_embeddings'] = reduced[idx:idx + n_embeddings]
        # Calculate distances in reduced space
        result['distances'] = [
            euclidean(result['reduced_embeddings'][0], emb)
            for emb in result['reduced_embeddings'][1:]
        ]
        idx += n_embeddings
    
    return all_results

def plot_results(results, snr_range, output_dir):
    """Plot and save the results, grouped by reference file."""
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
        output_name = f"noise_impact_{ref_name}"
        
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
    
    # Create timestamped run directory
    run_dir = create_run_directory(args.output)
    print(f"Saving results to: {run_dir}")
    
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
    
    if args.verbose:
        print(f"\nFound {len(reference_files)} reference files and {len(noise_files)} noise files")
        print(f"Will process {len(reference_files) * len(noise_files)} combinations")
        print(f"Total embeddings: {len(reference_files) * len(noise_files) * (args.snr_steps + 1)}")
    
    # Collect all embeddings
    all_results = []
    for ref_file in reference_files:
        for noise_file in noise_files:
            result = collect_embeddings(
                ref_file, noise_file, snr_range, model, args.verbose
            )
            all_results.append(result)
    
    # Perform dimensionality reduction on all embeddings together
    all_results = reduce_embeddings(all_results, args.reduced_dim, args.verbose)
    
    # Plot and save results
    plot_results(all_results, snr_range, run_dir)
    print(f"Processing complete. Results saved in {run_dir}")

if __name__ == "__main__":
    main()