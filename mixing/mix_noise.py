import torch
import torchaudio
import argparse
from pathlib import Path
from datetime import datetime
import json

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

def save_mixed_audio(output_dir, clean_file, noise_file, noisy_signal, snr_db, sample_rate=32000):
    """Save mixed audio and its metadata."""
    clean_name = Path(clean_file).stem
    noise_name = Path(noise_file).stem
    
    # Create filename with SNR information
    output_filename = f"{clean_name}_mixed_with_{noise_name}_snr{snr_db:+.1f}dB.wav"
    output_path = output_dir / output_filename
    
    # Save audio file
    torchaudio.save(output_path, noisy_signal, sample_rate)
    
    # Create metadata
    metadata = {
        "reference_file": str(clean_file),
        "noise_file": str(noise_file),
        "snr_db": int(snr_db),
        "sample_rate": int(sample_rate),
        "timestamp": datetime.now().isoformat(),
        "output_file": str(output_filename)
    }
    
    # Save metadata
    metadata_path = output_dir / f"{output_filename}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Mix audio files with noise at specified SNR levels")
    parser.add_argument('reference', help='Path to reference audio file or directory')
    parser.add_argument('noise', help='Path to noise file or directory')
    parser.add_argument('--min-snr', type=float, default=-40, help='Minimum SNR in dB')
    parser.add_argument('--max-snr', type=float, default=40, help='Maximum SNR in dB')
    parser.add_argument('--snr-steps', type=int, default=17, help='Number of SNR steps')
    parser.add_argument('--output', default='.', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()

def process_file_pair(clean_file, noise_file, snr_range, output_dir, verbose=False):
    """Process a single pair of reference and noise files."""
    if verbose:
        print(f"\nProcessing:\nReference: {clean_file}\nNoise: {noise_file}")
    
    # Load audio files
    clean_wave, clean_length = load_and_normalize_audio(clean_file)
    noise_wave, noise_length = load_and_normalize_audio(noise_file)
    
    if verbose:
        print(f"Clean audio length: {clean_length/32000:.2f}s")
        print(f"Noise audio length: {noise_length/32000:.2f}s")
    
    # Prepare audio to same length
    clean_wave, noise_wave, used_length = prepare_audio_pair(
        clean_wave, noise_wave, clean_length, noise_length)
    
    if verbose:
        print(f"Using length: {used_length/32000:.2f}s")
    
    # Mix and save for all SNR levels
    for snr in snr_range:
        if verbose:
            print(f"Processing SNR: {snr}dB")
        
        noisy_wave = mix_noise(clean_wave, noise_wave, snr)
        save_mixed_audio(output_dir, clean_file, noise_file, noisy_wave, snr)

def main():
    args = parse_arguments()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup experiment parameters
    snr_range = torch.linspace(args.min_snr, args.max_snr, args.snr_steps)
    
    # Get file lists
    reference_path = Path(args.reference)
    noise_path = Path(args.noise)
    
    reference_files = [reference_path] if reference_path.is_file() else list(reference_path.glob('*.wav'))
    noise_files = [noise_path] if noise_path.is_file() else list(noise_path.glob('*.wav'))
    
    if not reference_files or not noise_files:
        raise ValueError(f"No WAV files found in specified paths")
    
    # Save experiment configuration
    config = {
        "timestamp": timestamp,
        "reference_files": [str(f) for f in reference_files],
        "noise_files": [str(f) for f in noise_files],
        "snr_range": {
            "min": float(args.min_snr),
            "max": float(args.max_snr),
            "steps": args.snr_steps
        }
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Process all combinations
    for ref_file in reference_files:
        for noise_file in noise_files:
            process_file_pair(ref_file, noise_file, snr_range, output_dir, args.verbose)
    
    print(f"Mixed audio files saved to {output_dir}")

if __name__ == "__main__":
    main()