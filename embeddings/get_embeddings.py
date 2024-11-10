import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiocraft_fork.audiocraft.models import MusicGen
from embeddings.state_manager import state_manager
from h5_processor import H5Manager  # Import the new H5Manager

import torchaudio
import numpy as np
import os
import argparse
from pathlib import Path

def process_file(file, model, method="last", device="cuda"):
    # Clear any previous state and set method
    state_manager.clear_embedding()
    state_manager.set_method(method)
    
    waveform, sample_rate = torchaudio.load(file)
    num_samples = waveform.shape[1]
    waveform = waveform.unsqueeze(0).to(device)
    duration = num_samples / sample_rate + 0.04 # guaruntee longer than waveform, but don't generate new tokens unnecessarily
    
    model.set_generation_params(duration=duration)  
    waveform = model.generate_continuation(waveform, sample_rate) #, progress=True)
    
    # Get the embedding that was captured during the transformer's forward pass
    embedding = state_manager.get_embedding()

    return embedding

def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for audio files")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("-s", "--segment", type=str, default=None, help="Segment length")
    parser.add_argument("-t", "--stride", type=str, default=None, help="Stride length")
    parser.add_argument("-m", "--method", default="last", choices=["last", "mean"], help="Embedding method to use")
    parser.add_argument("-o", "--override", action="store_true", help="Override existing embedding file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Prepare paths
    if args.segment is None and args.stride is None:
        data_path = f"data/{args.dataset}/raw/"
        output_dir = os.path.join("embeddings", args.dataset, "raw")
    else:
        segment = args.segment if args.segment is not None else "all"
        stride = args.stride if args.stride is not None else "none"
        data_path = f"data/{args.dataset}/s{segment}-t{stride}/"
        output_dir = os.path.join("embeddings", args.dataset, f"s{segment}-t{stride}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.method}_embeddings.h5")

    return args, data_path, output_file

def process_directory(directory, h5_manager, model, method, verbose=False):
    """Recursively process all WAV files in a directory and its subdirectories."""
    processed_files = h5_manager.get_processed_files()
    
    for item in os.scandir(directory):
        if item.is_file() and item.name.endswith('.wav'):
            if str(item.path) in processed_files:
                if verbose:
                    print(f"Skipping already processed file: {item.path}")
                continue
            
            try:
                if verbose:
                    print(f"Processing: {item.path}")
                embedding = process_file(item.path, model, method=method).numpy()
                h5_manager.append_embedding(item.path, embedding)
            except Exception as e:
                print(f"Error processing {item.path}: {e}")
                continue
                
        elif item.is_dir():
            process_directory(item.path, h5_manager, model, method, verbose)

def main():
    args, data_path, output_file = parse_args()

    if not os.path.exists(data_path):
        print(f"Could not find data folder {data_path}")
        raise NotImplementedError
    
    if args.override and os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"Removed existing file: {output_file}")
        except OSError as e:
            print(f"Error removing file: {e}")

    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    print("Successfully Loaded Model")

    h5_manager = H5Manager(output_file, data_path)
    
    try:
        process_directory(data_path, h5_manager, model, args.method, args.verbose)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Progress has been saved.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        processed_files = h5_manager.get_processed_files()
        print(f"\nTotal files processed: {len(processed_files)}")
        print(f"Results saved to: {output_file}")
