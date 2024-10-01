import torch
from torch import nn
import torchaudio
from audiocraft.models import MusicGen
import json
from tqdm import tqdm
import os
import click

def load_processed_files(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files, log_file):
    with open(log_file, 'w') as f:
        json.dump(list(processed_files), f)

def enhanced_preprocess_and_encode(filename, model, device='cuda'):
    waveform, sample_rate = torchaudio.load(filename)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary (MusicGen expects 32kHz)
    if sample_rate != 32000:
        print(f"Received a sample rate of {sample_rate}. Resampling to 32kHz...")
        waveform = torchaudio.transforms.Resample(sample_rate, 32000)(waveform)
    
    # Normalize waveform
    waveform = waveform / waveform.abs().max()
    
    # Ensure correct shape and device
    waveform = waveform.unsqueeze(0).to(device)
    
    # Print waveform statistics
    print(f"Waveform shape: {waveform.shape}")
    print(f"Waveform stats - Min: {waveform.min().item():.4f}, Max: {waveform.max().item():.4f}")
    print(f"Mean: {waveform.mean().item():.4f}, Std: {waveform.std().item():.4f}")
    
    # Encode
    with torch.no_grad():
        encoded_frames = model.compression_model.encode(waveform)
    
    codes = encoded_frames[0]  # [B, K, T]
    
    # Print encoding statistics
    print(f"\nEncoded shape: {codes.shape}")
    for k in range(codes.shape[1]):
        unique_codes = torch.unique(codes[0, k])
        print(f"Codebook {k}:")
        print(f"  Unique tokens: {len(unique_codes)}")
        print(f"  Min: {codes[0, k].min().item()}, Max: {codes[0, k].max().item()}")
        print(f"  Mean: {codes[0, k].float().mean().item():.2f}, Std: {codes[0, k].float().std().item():.2f}")
    
    return codes

def process_file(file, model, method="last", device="cuda"):
    codes = enhanced_preprocess_and_encode(file, model, device)
    
    gen_sequence = get_patterns(model, codes)
    
    print(f"\nProcessing file: {file}")
    print(f"Generated sequence shape: {gen_sequence.shape}")
    
    x = prep_input(gen_sequence)
        
    with torch.no_grad():
        for i, layer in enumerate(model.lm.transformer.layers):
            x = x.half()
            x = layer(x)
            print(f"\nAfter layer {i}:")
            print(f"  Shape: {x.shape}")
            print(f"  Min: {x.min().item():.4f}, Max: {x.max().item():.4f}")
            print(f"  Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")

    if method == "last":
        final_embedding = x[:,-1:,:].cpu().flatten().data.numpy().tolist()
    elif method == "mean":
        final_embedding = x.mean(axis=1).cpu().flatten().data.numpy().tolist()
    else:
        print("Invalid embedding capture method")
        raise ValueError
    
    print("\nFinal embedding:")
    print(f"  Shape: {len(final_embedding)}")
    print(f"  Min: {min(final_embedding):.4f}, Max: {max(final_embedding):.4f}")
    print(f"  Mean: {sum(final_embedding)/len(final_embedding):.4f}")
    
    return final_embedding


def preprocess_waveform(filename, device='cuda'):
    if type(filename) == str:
        waveform, sample_rate = torchaudio.load(filename)
    else:
        print("DuBose make this function work with a whole list")
        raise NotImplementedError

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform.unsqueeze(0).to(device)


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
    # retrieve the start_offset in the sequence:
    # it is the first sequence step that contains the `start_offset` timestep
    start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
    assert start_offset_sequence is not None

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

    # Diagnostic information
    token_stats = []
    unique_tokens = []

    # Apply each embedding layer to its corresponding codebook and sum the results
    embedded = []
    for k in range(K):
        # Replace -1 with the last index in vocab (which will be mapped to zero vector due to padding_idx)
        seq_k = torch.where(sequence[:, k] == pad_token, torch.tensor(vocab_size - 1, device=device), sequence[:, k])
        emb_k = emb[k](seq_k)
        embedded.append(emb_k)
        
        # Collect diagnostic information
        unique_k = torch.unique(seq_k)
        token_stats.append({
            'min': seq_k.min().item(),
            'max': seq_k.max().item(),
            'mean': seq_k.float().mean().item(),
            'std': seq_k.float().std().item(),
            'num_unique': len(unique_k)
        })
        unique_tokens.append(unique_k.cpu().numpy())
    
    input_ = sum(embedded)
    
    # Print diagnostic information
    print("\nTokenization Diagnostics:")
    for k in range(K):
        print(f"Codebook {k}:")
        print(f"  Min: {token_stats[k]['min']}, Max: {token_stats[k]['max']}")
        print(f"  Mean: {token_stats[k]['mean']:.2f}, Std: {token_stats[k]['std']:.2f}")
        print(f"  Unique tokens: {token_stats[k]['num_unique']}")
        print(f"  Sample of unique tokens: {unique_tokens[k][:10]}...")
    
    print(f"\nFinal input shape: {input_.shape}")
    print(f"Input statistics - Min: {input_.min().item():.4f}, Max: {input_.max().item():.4f}")
    print(f"Mean: {input_.mean().item():.4f}, Std: {input_.std().item():.4f}")
    
    return input_

@click.command()
@click.argument("dataset")
@click.option("-s", "--segment", default=None)
@click.option("-t", "--stride", default=None)
@click.option("-m", "--method", default="last", help="Embedding method to use")

def main(dataset, method, segment, stride):
    # Parse and prep args
    if segment is None and stride is None: # use raw audio
        data_path = f"../data/{dataset}/raw/"
        log_file = os.path.join(dataset, "raw", f"{method}_processed_files.json")
        output_file = os.path.join(dataset, "raw", f"{method}_embeddings.json")
        os.makedirs(os.path.join(dataset, "raw"), exist_ok=True)

    else: # use cut audio
        segment = segment if segment is not None else "all"
        stride = stride if stride is not None else "none"

        data_path = f"../data/{dataset}/s{segment}-t{stride}/"
        log_file = os.path.join(dataset, f"s{segment}-t{stride}", f"{method}_processed_files.json")
        output_file = os.path.join(dataset, f"s{segment}-t{stride}", f"{method}_embeddings.json")
        os.makedirs(os.path.join(dataset, f"s{segment}-t{stride}"), exist_ok=True)

    if not os.path.exists(data_path):
        print("Could not find data folder " + data_path)
        raise NotImplementedError


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
            # if full_path in processed_files or ".wav" not in full_path:
            #     continue
            
            embedding = process_file(full_path, model, method=method)
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