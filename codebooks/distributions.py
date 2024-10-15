import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import yaml
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

def format_results(jsd_per_codebook, cos_sim_per_codebook, dataset_names):   
    print(f"Comparison between {dataset_names[0]} and {dataset_names[1]}:")
    print("Jensen-Shannon Divergence per Codebook:")
    for i, jsd in enumerate(jsd_per_codebook):
        print(f"  Codebook {i+1}: {jsd:.4f}")
    print("\nCosine Similarity per Codebook:")
    for i, cos_sim in enumerate(cos_sim_per_codebook):
        print(f"  Codebook {i+1}: {cos_sim:.4f}")

def jensen_shannon_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def cosine_similarity(p, q):
    return 1 - cosine(p, q)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_codebooks(filename):
    with h5py.File(filename, 'r') as f:
        if 'codebooks' not in f:
            raise ValueError(f"No 'codebooks' group found in {filename}")
        
        codebooks_group = f['codebooks']
        codebooks = []
        for key in codebooks_group.keys():
            codebooks.append(codebooks_group[key][()][0]) # codebooks come in w/ batch dimension
    
    if not codebooks:
        raise ValueError(f"No codebooks found in {filename}")
    
    return codebooks  # Return as a list of arrays

def get_filenames(config):
    files = []
    for data in config['datasets'][:2]:  # Only use the first two datasets
        sampling = f"s{data['segment']}-t{data['stride']}" if data['segment'] and data['stride'] else "raw"
        path = os.path.join(data['dataset'], sampling)
        if os.path.exists(path):
            filename = os.path.join(path, "codebooks.h5")
            files.append(filename)
        else:
            print(f"Invalid file path specified: {path}")

    if len(files) != 2:
        raise ValueError("Exactly two valid datasets are required")
    return files

def compare_distributions(dataset1, dataset2):
    assert len(dataset1[0]) == len(dataset2[0])
    n_codebooks = len(dataset1[0])

    jsd_per_codebook = []
    cos_sim_per_codebook = []
    
    for i in range(n_codebooks):
        # Flatten all codebooks for this position
        flat1 = np.concatenate([cb[i].flatten() for cb in dataset1])
        flat2 = np.concatenate([cb[i].flatten() for cb in dataset2])
        
        # Compute frequencies
        max_value = max(flat1.max(), flat2.max())
        freq1 = np.bincount(flat1, minlength=max_value+1)
        freq2 = np.bincount(flat2, minlength=max_value+1)
        
        # Normalize to get probability distributions
        p1 = freq1 / freq1.sum()
        p2 = freq2 / freq2.sum()
        
        jsd_per_codebook.append(jensen_shannon_divergence(p1, p2))
        cos_sim_per_codebook.append(cosine_similarity(p1, p2))
    
    return jsd_per_codebook, cos_sim_per_codebook

def main(config_file):
    config = load_config(config_file)
    files = get_filenames(config)
    
    datasets = [load_codebooks(file) for file in files]    
    jsd_per_codebook, cos_sim_per_codebook = compare_distributions(datasets[0], datasets[1])
    
    dataset_names = [data['dataset'] for data in config['datasets'][:2]]
    format_results(jsd_per_codebook, cos_sim_per_codebook, dataset_names)

if __name__ == "__main__":
    main('config.yaml')