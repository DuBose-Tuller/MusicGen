from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import h5py
import json
import os
import yaml
import argparse
from datetime import datetime
import hashlib

EMBEDDING_SIZE = 1536

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Classification of embeddings")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to process')
    parser.add_argument('--output', help='Output filename for the results')
    parser.add_argument('--verbose', '-v', action="store_true", help="Extra print statements for debugging")
    return parser.parse_args()

def merge_config(file_config, args):
    if args.datasets:
        file_config['datasets'] = [d for d in file_config['datasets'] if d['dataset'] in args.datasets]
    return file_config

def process_h5(file):
    """Load embeddings from H5 file."""
    embeddings = []
    with h5py.File(file, 'r') as f:
        emb_group = f['embeddings']
        for name in emb_group:
            embeddings.append(emb_group[name][()])
    return np.array(embeddings)

def get_filenames(config):
    files = []
    for data in config['datasets']:
        if data['segment'] and data['stride']:
            sampling = f"s{data['segment']}-t{data['stride']}"
        else:
            sampling = "raw"
        path = os.path.join(data['dataset'], sampling)
        if os.path.exists(path):
            filename = os.path.join(path, f"{data['method']}_embeddings.h5")
            files.append(filename)       

    if not files:
        raise ValueError("No valid files found")

    return files

def construct_dataset(sources, verbose=False):
    X_arrays = []
    y_arrays = []
    
    for i, source in enumerate(sources):
        embeddings = process_h5(source)
        if verbose:
            print(f"Dataset {i}: {embeddings.shape[0]} samples")
        labels = np.full((embeddings.shape[0],), i)
        X_arrays.append(embeddings)
        y_arrays.append(labels)

    X = np.concatenate(X_arrays, axis=0)
    y = np.concatenate(y_arrays, axis=None)
    
    if verbose:
        for i in range(len(sources)):
            print(f"Class {i} count: {np.sum(y == i)}")
    
    return X, y

def multiclass_model(X, y, verbose=False):
    # Print initial class distribution
    if verbose:
        print("\nInitial class distribution:")
        for i in range(len(np.unique(y))):
            print(f"Class {i}: {np.sum(y == i)} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        tol=1e-6,
        C=0.1
    ).fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    # Print prediction distribution
    if verbose:
        print("\nPrediction distribution:")
        for i in range(len(np.unique(y))):
            print(f"Class {i}: {np.sum(y_pred == i)} predictions")
    
    matrix = confusion_matrix(y_test, y_pred)
    metrics = {
        "f1": f1_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }
    
    # Add probability scores to metrics
    metrics["confidence"] = {
        "mean": np.mean(np.max(y_prob, axis=1)),
        "std": np.std(np.max(y_prob, axis=1))
    }
    
    # Print detailed per-class metrics if verbose
    if verbose:
        print("\nPer-class metrics:")
        for i in range(len(np.unique(y))):
            print(f"\nClass {i}:")
            print(f"Precision: {precision_score(y_test, y_pred, average=None)[i]:.3f}")
            print(f"Recall: {recall_score(y_test, y_pred, average=None)[i]:.3f}")
            print(f"F1: {f1_score(y_test, y_pred, average=None)[i]:.3f}")
    
    return matrix, metrics

def generate_output_filename(config):
    # Create a unique identifier based on the configuration
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the filename
    filename = f"classifier_{timestamp}_{config_hash}.json"
    
    return filename

def analyze_feature_space(X, y):
    """Analyze the distribution of features between classes."""
    
    # Basic statistics per class
    for i in range(len(np.unique(y))):
        X_class = X[y == i]
        print(f"\nClass {i} statistics:")
        print(f"Mean magnitude: {np.linalg.norm(X_class, axis=1).mean():.3f}")
        print(f"Std magnitude: {np.linalg.norm(X_class, axis=1).std():.3f}")
        print(f"Mean: {X_class.mean():.3f}")
        print(f"Std: {X_class.std():.3f}")
        print(f"Min: {X_class.min():.3f}")
        print(f"Max: {X_class.max():.3f}")
    
    # Feature-wise statistics
    feature_diffs = []
    for feat in range(X.shape[1]):
        class_0_mean = X[y == 0, feat].mean()
        class_1_mean = X[y == 1, feat].mean()
        diff = abs(class_0_mean - class_1_mean)
        feature_diffs.append((feat, diff))
    
    # Sort features by difference between classes
    feature_diffs.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 most different features between classes:")
    for feat, diff in feature_diffs[:10]:
        print(f"Feature {feat}: {diff:.3f} difference")

def save_results(config, matrix, metrics, output_path):
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "confusion_matrix": matrix.tolist(),
        "metrics": metrics,
        "dataset_names": [d['dataset'] for d in config['datasets']]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Parse arguments and load config
    args = parse_arguments()
    file_config = load_config(args.config)
    config = merge_config(file_config, args)

    # Create output directory
    output_dir = os.path.join("../results", "classifier")
    os.makedirs(output_dir, exist_ok=True)

    # Get files and process
    files = get_filenames(config)
    print("Processing files:", files)
    
    X, y = construct_dataset(files, verbose=args.verbose)
    if args.verbose:
        analyze_feature_space(X, y)
    cm, metrics = multiclass_model(X, y, verbose=args.verbose)

    # Save results
    output_filename = generate_output_filename(config)
    output_path = os.path.join(output_dir, output_filename)
    save_results(config, cm, metrics, output_path)
    print(f"\nResults have been saved to '{output_path}'")

if __name__ == "__main__":
    main()