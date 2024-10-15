import torch
import torchaudio
from audiocraft.models import MusicGen
from tqdm import tqdm
import os
import argparse
import h5py
import numpy as np

def save_processed_files(processed_files, log_file):
    with h5py.File(log_file, 'a') as f:
        if 'processed_files' in f:
            del f['processed_files']  # Delete existing dataset
        f.create_dataset('processed_files', data=np.array(list(processed_files), dtype=h5py.special_dtype(vlen=str)))

def load_processed_files(log_file):
    if os.path.exists(log_file):
        with h5py.File(log_file, 'r') as f:
            if 'processed_files' in f:
                return set(f['processed_files'][()])
    return set()

def process_file(file, model, device="cuda"):
    codes = preprocess_waveform(file, model, device)
    gen_sequence = get_patterns(model, codes, device)

    return gen_sequence.cpu().numpy()


def preprocess_waveform(filename, model, device='cuda'):
    if type(filename) == str:
        waveform, sample_rate = torchaudio.load(filename)
    else:
        print("DuBose make this function work with a whole list")
        raise NotImplementedError

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary (MusicGen expects 32kHz)
    if sample_rate != 32000:
        #print(f"Sample rate is {sample_rate}, resampling to 32kHz...")
        waveform = torchaudio.transforms.Resample(sample_rate, 32000)(waveform)
    
    # Ensure correct shape and device
    waveform = waveform.unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        encoded_frames = model.compression_model.encode(waveform)
    
    codes = encoded_frames[0]  # [B, K, T]
    return codes


# Largely ripped from LMModel.generate() in lm.py
def get_patterns(model, prompt, device="cuda"):
    B, K, T = prompt.shape
    start_offset = T

    pattern = model.lm.pattern_provider.get_pattern(T)

    # this token is used as default value for codes that are not generated yet
    unknown_token = -1

    # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
    gen_codes = torch.full((B, K, T), unknown_token, dtype=torch.long, device=device)
    # filling the gen_codes with the prompt if needed
    gen_codes[..., :start_offset] = prompt
    # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
    gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, 2048)

    return gen_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Generate codebooks for audio files")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("-s", "--segment", type=str, default=None, help="Segment length")
    parser.add_argument("-t", "--stride", type=str, default=None, help="Stride length")
    parser.add_argument("-o", "--override", action="store_true", help="Override existing codebook file")

    args = parser.parse_args()

    # Prepare args
    if args.segment is None and args.stride is None:
        data_path = f"../data/{args.dataset}/raw/"
        log_file = os.path.join(args.dataset, "raw", "codebooks_processed.h5")
        output_file = os.path.join(args.dataset, "raw", "codebooks.h5")
        os.makedirs(os.path.join(args.dataset, "raw"), exist_ok=True)
    else:
        segment = args.segment if args.segment is not None else "all"
        stride = args.stride if args.stride is not None else "none"

        data_path = f"../data/{args.dataset}/s{segment}-t{stride}/"
        log_file = os.path.join(args.dataset, f"s{segment}-t{stride}", "codebooks_processed.h5")
        output_file = os.path.join(args.dataset, f"s{segment}-t{stride}", "codebooks.h5")
        os.makedirs(os.path.join(args.dataset, f"s{segment}-t{stride}"), exist_ok=True)

    return args, data_path, output_file, log_file

def main(): 
    args, data_path, output_file, log_file = parse_args()

    if not os.path.exists(data_path):
        print("Could not find data folder " + data_path)
        raise NotImplementedError
    
    if args.override:
        try:
            os.remove(log_file)
            os.remove(output_file)
        except FileNotFoundError:
            print("Tried to override, but file not found! Ignoring...")
            pass

    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    print("Successfully Loaded Model")

    processed_files = load_processed_files(log_file)
    print(f"Found {len(processed_files)} processed files.")

    try:
        with h5py.File(output_file, 'a') as f:
            if 'codebooks' not in f:
                codebooks_group = f.create_group('codebooks')
            else:
                codebooks_group = f['codebooks']
            
            for file in tqdm(os.listdir(data_path)):
                full_path = os.path.join(data_path, file)
                if full_path in processed_files or ".wav" not in full_path:
                    continue
                
                codebook = process_file(full_path, model)
                
                # Use a sanitized version of the full path as the dataset name
                dataset_name = full_path.replace('/', '_').replace('\\', '_')
                
                # Check if dataset already exists
                if dataset_name in codebooks_group:
                    del codebooks_group[dataset_name]  # Delete existing dataset
                
                # Create new dataset
                codebooks_group.create_dataset(dataset_name, data=codebook, compression="gzip")
                
                processed_files.add(full_path)
                
                # Save progress
                if len(processed_files) % 100 == 0:
                    save_processed_files(processed_files, log_file)

    except KeyboardInterrupt:
        print("Process interrupted. Saving progress...")

    finally:
        # Save final progress
        save_processed_files(processed_files, log_file)

    print(f"Processed {len(processed_files)} files. Results saved to {output_file}")

if __name__ == "__main__":
    main()
