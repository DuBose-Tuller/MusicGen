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

def save_progress(h5_file, embeddings_dict, processed_files):
    """Save both embeddings and processed files to HDF5 file."""
    with h5py.File(h5_file, 'w') as f:
        # Save embeddings
        emb_group = f.create_group('embeddings')
        for filename, embedding in embeddings_dict.items():
            emb_group.create_dataset(filename, data=np.array(embedding, dtype=np.float32))
        
        # Save processed files list as an attribute
        f.attrs['processed_files'] = list(processed_files)

def process_file(file, model, method="last", device="cuda"):
    # Clear any previous state
    state_manager.clear_embedding()
    
    waveform, sample_rate = torchaudio.load(file)
    waveform = waveform.unsqueeze(0).to(device)
    
    waveform = model.generate_continuation(waveform, sample_rate)
    
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
        output_dir = os.path.join(args.dataset, "raw")
    else:
        segment = args.segment if args.segment is not None else "all"
        stride = args.stride if args.stride is not None else "none"
        data_path = f"data/{args.dataset}/s{segment}-t{stride}/"
        output_dir = os.path.join(args.dataset, f"s{segment}-t{stride}")

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
    model.set_generation_params(duration=duration + 0.04)   

    processed_files = load_processed_files(output_file)
    embeddings = {}

    # Load existing embeddings if the file exists
    if os.path.exists(output_file):
        with h5py.File(output_file, 'r') as f:
            emb_group = f['embeddings']
            for filename in emb_group:
                embeddings[filename] = emb_group[filename][()]

    try:
        for file in tqdm(os.listdir(data_path)):
            full_path = os.path.join(data_path, file)
            if full_path in processed_files or ".wav" not in full_path:
                continue
            
            embedding = process_file(full_path, model, method=args.method)
            embedding = embedding.numpy()
            embeddings[full_path] = embedding
            processed_files.add(full_path)
            
            # Save progress periodically
            if len(processed_files) % 100 == 0:
                save_progress(output_file, embeddings, processed_files)

    except KeyboardInterrupt:
        print("Process interrupted. Saving progress...")

    finally:
        # Save final progress
        save_progress(output_file, embeddings, processed_files)

    print(f"Processed {len(processed_files)} files. Results saved to {output_file}")