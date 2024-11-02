from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import h5py
import json
import os
import yaml
import argparse
from datetime import datetime
import hashlib
from pathlib import Path

EMBEDDING_SIZE = 1536

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Classification of embeddings by instrument")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output', help='Output filename for the results')
    parser.add_argument('--verbose', '-v', action="store_true", help="Extra print statements for debugging")
    return parser.parse_args()

def get_data_filenames(dataset_path, segment=None, stride=None):
    """Get the path to the data files directory."""
    if segment is None and stride is None:
        data_path = os.path.join("data", dataset_path, "raw")
    else:
        segment = segment if segment is not None else "all"
        stride = stride if stride is not None else "none"
        data_path = os.path.join("data", dataset_path, f"s{segment}-t{stride}")
    
    return data_path

def extract_instrument_from_path(file_path, pattern):
    """Extract instrument name using regex pattern."""
    import re
    match = re.search(pattern, file_path)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract instrument from path {file_path} using pattern {pattern}")

def get_class_names_from_data(data_path, pattern):
    """Extract unique class names from the data directory structure."""
    class_names = set()
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):  # only process wav files
                full_path = os.path.join(root, file)
                try:
                    instrument = extract_instrument_from_path(full_path, pattern)
                    class_names.add(instrument)
                except ValueError as e:
                    print(f"Warning: {e}")
                    continue
    return sorted(list(class_names))

def process_h5_with_instruments(h5_file, data_path, pattern, verbose=False):
    """Load embeddings from H5 file with instrument labels."""
    embeddings = []
    instruments = []
    
    with h5py.File(h5_file, 'r') as f:
        emb_group = f['embeddings']
        for name in emb_group:
            try:
                instrument = extract_instrument_from_path(name, pattern)
                embeddings.append(emb_group[name][()])
                instruments.append(instrument)
            except ValueError as e:
                if verbose:
                    print(f"Warning: {e}")
                continue
            
    if verbose:
        unique_instruments = set(instruments)
        print(f"\nFound {len(unique_instruments)} instruments:")
        for inst in sorted(unique_instruments):
            count = instruments.count(inst)
            print(f"{inst}: {count} samples")
    
    return np.array(embeddings), instruments

def construct_instrument_dataset(h5_path, data_path, pattern, verbose=False):
    """Construct dataset with instrument labels."""
    if verbose:
        print(f"Processing embeddings from: {h5_path}")
        print(f"Using data path: {data_path}")
        print(f"Using pattern: {pattern}")
    
    # Get all possible class names from the data directory
    class_names = get_class_names_from_data(data_path, pattern)
    if verbose:
        print(f"\nFound {len(class_names)} unique instruments in data directory:")
        print(", ".join(class_names))
    
    # Process the H5 file
    X, instruments = process_h5_with_instruments(h5_path, data_path, pattern, verbose)
    
    # Convert instrument labels to numeric classes
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)  # Fit on all possible classes first
    y = label_encoder.transform(instruments)
    
    if verbose:
        print("\nClass mapping:")
        for i, instrument in enumerate(label_encoder.classes_):
            print(f"Class {i}: {instrument}")
    
    return X, y, label_encoder.classes_

def compute_metrics(y_test, y_pred, y_prob, is_binary=False):
    """Compute classification metrics, handling both binary and multi-class cases."""
    metrics = {
        "f1": f1_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='macro')
    }

    # Only compute ROC AUC for binary classification
    if is_binary:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
        except Exception:
            metrics["roc_auc"] = None
    
    # Add probability scores to metrics
    metrics["confidence"] = {
        "mean": float(np.mean(np.max(y_prob, axis=1))),
        "std": float(np.std(np.max(y_prob, axis=1)))
    }
    
    return metrics

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
        C=0.1,
        solver='lbfgs'
    ).fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    # Print prediction distribution
    if verbose:
        print("\nPrediction distribution:")
        for i in range(len(np.unique(y))):
            print(f"Class {i}: {np.sum(y_pred == i)} predictions")
    
    matrix = confusion_matrix(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred, y_prob, is_binary=len(np.unique(y)) == 2)
    
    # Print detailed per-class metrics if verbose
    if verbose:
        print("\nPer-class metrics:")
        for i in range(len(np.unique(y))):
            print(f"\nClass {i}:")
            print(f"Precision: {precision_score(y_test, y_pred, average=None)[i]:.3f}")
            print(f"Recall: {recall_score(y_test, y_pred, average=None)[i]:.3f}")
            print(f"F1: {f1_score(y_test, y_pred, average=None)[i]:.3f}")
    
    return matrix, metrics

def save_results(config, matrix, metrics, class_names, output_path):
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "confusion_matrix": matrix.tolist(),
        "metrics": metrics,
        "class_names": class_names.tolist() if isinstance(class_names, np.ndarray) else list(class_names)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Parse arguments and load config
    args = parse_arguments()
    config = load_config(args.config)
    
    # Extract relevant configuration
    dataset_config = config['datasets'][0]  # Use first dataset
    pattern = config.get('instrument_pattern', r'^([^_]+)')  # Default pattern if not specified
    
    # Create output directory
    output_dir = os.path.join("../results", "classifier")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paths
    data_path = get_data_filenames(
        dataset_config['dataset'],
        dataset_config.get('segment'),
        dataset_config.get('stride')
    )
    
    h5_path = os.path.join(
        dataset_config['dataset'],
        f"s{dataset_config['segment']}-t{dataset_config['stride']}" if dataset_config['segment'] and dataset_config['stride'] else "raw",
        f"{dataset_config['method']}_embeddings.h5"
    )
    
    # Process dataset
    X, y, class_names = construct_instrument_dataset(
        h5_path,
        data_path,
        pattern,
        verbose=args.verbose
    )
    
    # Train and evaluate model
    cm, metrics = multiclass_model(X, y, verbose=args.verbose)
    
    # Generate output filename and save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    output_filename = f"instrument_classifier_{timestamp}_{config_hash}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    save_results(config, cm, metrics, class_names, output_path)
    print(f"\nResults have been saved to '{output_path}'")

if __name__ == "__main__":
    main()