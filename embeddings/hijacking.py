import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiocraft_fork.audiocraft.models import MusicGen
from embeddings.state_manager import state_manager

# import torch
# from torch import nn
import torchaudio
import json
from tqdm import tqdm
import os
import argparse

def load_processed_files(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files, log_file):
    with open(log_file, 'w') as f:
        json.dump(list(processed_files), f)

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

    # Prepare args
    if args.segment is None and args.stride is None:
        data_path = f"data/{args.dataset}/raw/"
        log_file = os.path.join("embeddings", args.dataset, "raw", f"{args.method}_processed_files.json")
        output_file = os.path.join("embeddings", args.dataset, "raw", f"{args.method}_embeddings.json")
        os.makedirs(os.path.join("embeddings", args.dataset, "raw"), exist_ok=True)
    else:
        segment = args.segment if args.segment is not None else "all"
        stride = args.stride if args.stride is not None else "none"

        data_path = f"data/{args.dataset}/s{segment}-t{stride}/"
        log_file = os.path.join("embeddings", args.dataset, f"s{segment}-t{stride}", f"{args.method}_processed_files.json")
        output_file = os.path.join("embeddings", args.dataset, f"s{segment}-t{stride}", f"{args.method}_embeddings.json")
        os.makedirs(os.path.join("embeddings", args.dataset, f"s{segment}-t{stride}"), exist_ok=True)

    return args, data_path, output_file, log_file

def main(): 
    args, data_path, output_file, log_file = parse_args()

    if not os.path.exists(data_path):
        print("Could not find data folder " + data_path)
        raise NotImplementedError
    
    if args.segment == None:
        duration = 15
    else:
        duration = float(args.segment)
    
    if args.override:
        try:
            os.remove(log_file)
            os.remove(output_file)
        except FileNotFoundError:
            print("Tried to override, but log file not found! Ignoring...")
            pass


    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    print("Succeessfully Loaded Model")
    model.set_generation_params(duration=duration + 0.04)   

    processed_files = load_processed_files(log_file)
    embeddings = {}

    # Load existing embeddings if the file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            embeddings = json.load(f)

    try:
        for file in tqdm(os.listdir(data_path)):
            full_path = os.path.join(data_path, file)
            if full_path in processed_files or ".wav" not in full_path:
                continue
            
            embedding = process_file(full_path, model, method=args.method)
            embedding = embedding.numpy().tolist()
            embeddings[full_path] = embedding
            processed_files.add(full_path)
            
            # Save progress
            if len(processed_files) % 100 == 0:
                save_processed_files(processed_files, log_file)
                with open(output_file, 'w') as f:
                    json.dump(embeddings, f)

    except KeyboardInterrupt:
        print("Process interrupted. Saving progress...")

    finally:
        # Save final progress
        save_processed_files(processed_files, log_file)
        with open(output_file, 'w') as f:
            json.dump(embeddings, f)

    print(f"Processed {len(processed_files)} files. Results saved to {output_file}")

# if __name__ == "__main__":
#     main()