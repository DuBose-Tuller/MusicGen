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
from pathlib import Path

class H5Manager:
    """Manager for H5 file operations with improved error handling and hierarchy support."""
    def __init__(self, h5_file, data_path):
        self.h5_file = h5_file
        self.data_path = data_path
        self.processed_files = self._load_processed_files()
        
    def _load_processed_files(self):
        """Load set of processed files from HDF5 file with improved error handling."""
        if not os.path.exists(self.h5_file):
            return set()
        
        try:
            with h5py.File(self.h5_file, 'r') as f:
                return set(f.attrs.get('processed_files', []))
        except (OSError, RuntimeError) as e:
            print(f"Warning: Could not read {self.h5_file}. Starting fresh. Error: {e}")
            return set()

    def _get_or_create_group(self, h5file, group_path):
        """Creates nested groups if they don't exist and returns the deepest group."""
        current_group = h5file
        for group_name in group_path:
            if group_name not in current_group:
                current_group = current_group.create_group(group_name)
            else:
                current_group = current_group[group_name]
        return current_group

    def append_embedding(self, filepath, embedding):
        """Append a single embedding to the HDF5 file with improved structure and error handling."""
        filepath = Path(filepath)
        try:
            relative_parts = filepath.relative_to(Path(self.data_path)).parts
        except ValueError:
            # If relative_to fails, just use the filename
            relative_parts = (filepath.name,)
            
        # Handle both cases: with subfolders and without
        if len(relative_parts) > 1:
            group_path = relative_parts[:-1]  # Use subfolder structure
            filename = relative_parts[-1]
        else:
            group_path = ()  # Empty tuple for root level
            filename = relative_parts[0]
        
        # Convert filename to a safe HDF5 dataset name
        safe_name = filename.replace('/', '_').replace('\\', '_')
        

        with h5py.File(self.h5_file, 'a') as f:
            # Create base embeddings group if it doesn't exist
            if 'embeddings' not in f:
                f.create_group('embeddings')
            
            # Get the proper group
            if group_path:
                # If we have subfolders, use the hierarchy
                current_group = self._get_or_create_group(f['embeddings'], group_path)
            else:
                # If no subfolders, use embeddings group directly
                current_group = f['embeddings']
            
            # Save the embedding
            if safe_name in current_group:
                del current_group[safe_name]  # Replace if exists
            current_group.create_dataset(safe_name, data=np.array(embedding, dtype=np.float32))
            
            # Update processed files list
            processed_files = set(f.attrs.get('processed_files', []))
            processed_files.add(str(filepath))
            f.attrs['processed_files'] = list(processed_files)
            
            # Update the instance's processed files
            self.processed_files = processed_files


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
    for item in tqdm(os.scandir(directory)):
        if item.is_file() and item.name.endswith('.wav'):
            if item.path in h5_manager.processed_files:
                if verbose:
                    print(f"Skipping already processed file: {item.path}")
                continue
                
            try:
                embedding = process_file(item.path, model, method=method).numpy()
                h5_manager.append_embedding(item.path, embedding)
                if verbose:
                    print(f"Processed: {item.path}")
            except Exception as e:
                print(f"Error processing {item.path}: {e}")
                
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
        total_processed = len(h5_manager.processed_files)
        print(f"\nTotal files processed: {total_processed}")
        print(f"Results saved to: {output_file}")
