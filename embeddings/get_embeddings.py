import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiocraft_fork.audiocraft.models import MusicGen
from embeddings.state_manager import state_manager

import torchaudio
import numpy as np
from tqdm import tqdm
import os
import argparse
import h5py

def load_processed_files(h5_file):
    """Load set of processed files from HDF5 file."""
    if not os.path.exists(h5_file):
        return set()
    
    try:
        with h5py.File(h5_file, 'r') as f:
            if 'processed_files' in f.attrs:
                return set(f.attrs['processed_files'])
    except OSError:
        print(f"Warning: Could not read {h5_file}. Starting fresh.")
    return set()

def append_embedding(h5_file, filename, embedding):
    """Append a single embedding to the HDF5 file."""
    # Convert filename to a safe HDF5 dataset name
    safe_name = filename.replace('/', '_').replace('\\', '_')
    
    # Open in append mode ('a')
    with h5py.File(h5_file, 'a') as f:
        # Create embeddings group if it doesn't exist
        if 'embeddings' not in f:
            f.create_group('embeddings')
            
        # Save the embedding
        f['embeddings'].create_dataset(safe_name, data=np.array(embedding, dtype=np.float32))
        
        # Update processed files list
        processed_files = set(f.attrs.get('processed_files', []))
        processed_files.add(filename)
        f.attrs['processed_files'] = list(processed_files)

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

def main():
    args, data_path, output_file = parse_args()

    if not os.path.exists(data_path):
        print(f"Could not find data folder {data_path}")
        raise NotImplementedError
    
    if args.segment == None:
        duration = 15
    else:
        duration = float(args.segment)
    
    if args.override:
        try:
            os.remove(output_file)
        except FileNotFoundError:
            print("Tried to override, but log file not found! Ignoring...")
            pass


    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    print("Succeessfully Loaded Model")   

    processed_files = load_processed_files(output_file)
    files_processed = 0

    try:
        for file in tqdm(os.listdir(data_path)):
            full_path = os.path.join(data_path, file)
            if full_path in processed_files or ".wav" not in full_path:
                continue
            
            embedding = process_file(full_path, model, method=args.method).numpy()
            
            # Append this embedding to the HDF5 file
            append_embedding(output_file, full_path, embedding)
            files_processed += 1

    except KeyboardInterrupt:
        print("\nProcess interrupted. Progress has been saved.")

    print(f"\nProcessed {files_processed} new files. All results saved to {output_file}")