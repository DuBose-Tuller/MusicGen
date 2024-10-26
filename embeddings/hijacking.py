import sys
import os
from queue import Queue, Empty
from threading import Thread, Lock
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiocraft_fork.audiocraft.models import MusicGen
from embeddings.state_manager import state_manager

import torchaudio
import numpy as np
from tqdm import tqdm
import os
import argparse
import h5py

class BackgroundWriter:
    def __init__(self, output_file):
        self.output_file = output_file
        self.queue = Queue()
        self.processed_files = set(load_processed_files(output_file))
        self.embeddings = {}
        self.lock = Lock()
        self.should_stop = False
        self.processed_count = 0
        
        # Load existing embeddings if file exists
        if os.path.exists(output_file):
            with h5py.File(output_file, 'r') as f:
                emb_group = f['embeddings']
                for filename in emb_group:
                    self.embeddings[filename] = emb_group[filename][()]
        
        # Start background thread
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def _process_queue(self):
        last_save_time = time.time()
        
        while not (self.should_stop and self.queue.empty()):
            try:
                # Wait for new data with timeout to check should_stop periodically
                data = self.queue.get(timeout=1.0)
                
                with self.lock:
                    filename, embedding = data
                    self.embeddings[filename] = embedding
                    self.processed_files.add(filename)
                    self.processed_count += 1
                
                # Save periodically (every 60 seconds) or if queue is empty
                current_time = time.time()
                if current_time - last_save_time > 60 or (self.should_stop and self.queue.empty()):
                    self._save_to_disk()
                    last_save_time = current_time
                
                self.queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in background thread: {str(e)}")
                continue

    def _save_to_disk(self):
        with self.lock:
            try:
                with h5py.File(self.output_file, 'w') as f:
                    # Save embeddings
                    emb_group = f.create_group('embeddings')
                    for filename, embedding in self.embeddings.items():
                        emb_group.create_dataset(filename, data=embedding)
                    
                    # Save processed files list
                    f.attrs['processed_files'] = list(self.processed_files)
            except Exception as e:
                print(f"Warning: Failed to save to disk: {e}")

    def add_result(self, filename: str, embedding: np.ndarray):
        """Add a new result to be written in the background."""
        self.queue.put((filename, embedding))

    def get_processed_files(self):
        """Thread-safe access to processed files set."""
        with self.lock:
            return set(self.processed_files)

    def get_queue_size(self):
        """Get current size of the queue."""
        return self.queue.qsize()

    def get_processed_count(self):
        """Get number of processed files."""
        with self.lock:
            return self.processed_count

    def shutdown(self):
        """Gracefully shutdown the background writer."""
        self.should_stop = True
        self.worker.join()
        self._save_to_disk()  # Final save

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
    
    if args.segment is None:
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

    # Initialize background writer
    writer = BackgroundWriter(output_file)

    try:
        # Get list of files to process
        files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
        processed_files = writer.get_processed_files()
        files_to_process = [f for f in files if os.path.join(data_path, f) not in processed_files]
        
        # Initialize progress bar for remaining files
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            last_processed_count = writer.get_processed_count()
            
            for file in files_to_process:
                full_path = os.path.join(data_path, file)
                
                # Process the file
                embedding = process_file(full_path, model, method=args.method)
                embedding = embedding.numpy()
                writer.add_result(full_path, embedding)
                
                # Update progress bar based on actual completed writes
                current_processed = writer.get_processed_count()
                pbar.update(current_processed - last_processed_count)
                last_processed_count = current_processed
                
                # Optionally add queue size to progress bar description
                pbar.set_description(f"Processing files (queue: {writer.get_queue_size()})")

    except KeyboardInterrupt:
        print("\nProcess interrupted. Waiting for final saves...")

    finally:
        # Ensure all writes are completed before exit
        writer.shutdown()
        print(f"\nProcessed files saved to {output_file}")