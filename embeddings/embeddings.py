import torch
from torch import nn
import torchaudio
from audiocraft.models import MusicGen
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
    codes = preprocess_waveform(file, model, device)

    gen_sequence = get_patterns(model, codes, device)
    x = prep_input(gen_sequence)
    
    del codes
    del gen_sequence
    
    with torch.no_grad():
        for layer in model.lm.transformer.layers:
            x = x.half()
            x = layer(x)

    if method == "last":
        final_embedding = x[:,-1:,:].cpu().flatten().data.numpy().tolist()
    
    elif method == "mean":
        final_embedding = x.mean(axis=1).cpu().flatten().data.numpy().tolist()

    else:
        print(f"Invalid embedding capture method {method}")
        raise ValueError
    
    return final_embedding


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
        print(f"Sample rate is {sample_rate}, resampling to 32kHz...")
        waveform = torchaudio.transforms.Resample(sample_rate, 32000)(waveform)
    
    # Normalize waveform
    waveform = waveform / waveform.abs().max()
    
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

class ScaledEmbedding(nn.Embedding):
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


def prep_input(sequence, pad_token=-1, embed_dim=1536, emb_lr=1.0):
    device = sequence.device
    B, K, S = sequence.shape
    
    # Adjust vocab_size to account for padding token and maximum value
    vocab_size = sequence.max().item() + 1
    emb = nn.ModuleList([ScaledEmbedding(vocab_size, embed_dim, padding_idx=pad_token, lr=emb_lr) for _ in range(K)]).to(device)

    # Apply each embedding layer to its corresponding codebook and sum the results
    embedded = []
    for k in range(K):
        # Replace -1 with the last index in vocab (which will be mapped to zero vector due to padding_idx)
        seq_k = torch.where(sequence[:, k] == pad_token, torch.tensor(vocab_size - 1, device=device), sequence[:, k])
        emb_k = emb[k](seq_k)
        embedded.append(emb_k)
    
    input_ = sum(embedded)
    return input_

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
        data_path = f"../data/{args.dataset}/raw/"
        log_file = os.path.join(args.dataset, "raw", f"{args.method}_processed_files.json")
        output_file = os.path.join(args.dataset, "raw", f"{args.method}_embeddings.json")
        os.makedirs(os.path.join(args.dataset, "raw"), exist_ok=True)
    else:
        segment = args.segment if args.segment is not None else "all"
        stride = args.stride if args.stride is not None else "none"

        data_path = f"../data/{args.dataset}/s{segment}-t{stride}/"
        log_file = os.path.join(args.dataset, f"s{segment}-t{stride}", f"{args.method}_processed_files.json")
        output_file = os.path.join(args.dataset, f"s{segment}-t{stride}", f"{args.method}_embeddings.json")
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
            print("Tried to override, but log file not found! Ignoring...")
            pass


    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    print("Succeessfully Loaded Model")

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

if __name__ == "__main__":
    main()