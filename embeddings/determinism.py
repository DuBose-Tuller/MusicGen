import torch
from audiocraft.models import MusicGen
import numpy as np
from embeddings import process_file
import os

# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# # Set random seeds
# torch.manual_seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)


def determinism_test(file_path, model, device="cuda", num_runs=5, tol=1e-5):
    embeddings = []
    for _ in range(num_runs):
        embedding = process_file(file_path, model, device=device)
        embeddings.append(np.array(embedding))
    
    for i in range(num_runs):
        COMPARE = 0
        is_close = np.allclose(embeddings[COMPARE], embeddings[i], rtol=tol, atol=tol)
        print(f"Run {COMPARE+1} and Run {i+1} are close within tolerance {tol}: {is_close}")
        if not is_close:
            diff = np.abs(embeddings[COMPARE] - embeddings[i])
            print(f"  Max difference: {np.max(diff)}")
            print(f"  Mean difference: {np.mean(diff)}")
            print(f"  Std of difference: {np.std(diff)}")
    
    return embeddings

# Load model
model = MusicGen.get_pretrained('facebook/musicgen-melody')

# Run the determinism test
file_path = "../data/troubleshoot/raw/high_tone_1309_phase_0.489.wav"
determinism_test(file_path, model, num_runs=2, device="cpu")
