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
    parser = argparse.ArgumentParser(description="Classification of embeddings")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to process')
    parser.add_argument('--output', help='Output filename for the results')
    parser.add_argument('--by-filename', action='store_true', help='Extract instrument from filename instead of folder structure')
    parser.add_argument('--instrument-pattern', help='Regex pattern to extract instrument from filename when using --by-filename')
    parser.add_argument('--verbose', '-v', action="store_true", help="Extra print statements for debugging")
    return parser.parse_args()

def merge_config(file_config, args):
    if args.datasets:
        file_config['datasets'] = [d for d in file_config['datasets'] if d['dataset'] in args.datasets]
    return file_config

def extract_instrument_from_path(file_path, by_filename=False, pattern=None):
    """Extract instrument name either from parent folder or filename."""
    path = Path(file_path)
    if by_filename:
        if pattern:
            import re
            match = re.search(pattern, path.name)
            if match:
                return match.group(1)
            else:
                raise ValueError(f"Could not extract instrument from filename {path.name} using pattern {pattern}")
        else:
            # Default behavior: split by underscore and take first part
            return path.stem.split('_')[0]
    else:
        # Extract from parent folder name
        return path.parent.name

def process_h5_with_instruments(file, by_filename=False, pattern=None, verbose=False):
    """Load embeddings from H5 file with instrument labels."""
    embeddings = []
    instruments = []
    
    with h5py.File(file, 'r') as f:
        emb_group = f['embeddings']
        for name in emb_group:
            instrument = extract_instrument_from_path(name, by_filename, pattern)

            if verbose:
                print(f"Found instrument: {instrument} from file {file}")

            embeddings.append(emb_group[name][()])
            instruments.append(instrument)
            
    if verbose:
        unique_instruments = set(instruments)
        print(f"\nFound {len(unique_instruments)} instruments:")
        for inst in sorted(unique_instruments):
            count = instruments.count(inst)
            print(f"{inst}: {count} samples")
    
    return np.array(embeddings), instruments

def construct_instrument_dataset(source, by_filename=False, pattern=None, verbose=False):
    """Construct dataset with instrument labels from a single source."""
    if verbose:
        print(source)
    
    X, instruments = process_h5_with_instruments(source, by_filename, pattern, verbose)
    
    # Convert instrument labels to numeric classes
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(instruments)
    
    if verbose:
        print("\nClass mapping:")
        for i, instrument in enumerate(label_encoder.classes_):
            print(f"Class {i}: {instrument}")
    
    return X, y, label_encoder.classes_

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
    n_classes = len(np.unique(y))
    
    # Basic statistics per class
    for i in range(n_classes):
        X_class = X[y == i]
        print(f"\nClass {i} statistics:")
        print(f"Mean magnitude: {np.linalg.norm(X_class, axis=1).mean():.3f}")
        print(f"Std magnitude: {np.linalg.norm(X_class, axis=1).std():.3f}")
        print(f"Mean: {X_class.mean():.3f}")
        print(f"Std: {X_class.std():.3f}")
        print(f"Min: {X_class.min():.3f}")
        print(f"Max: {X_class.max():.3f}")
    
    # Feature-wise statistics (only for binary classification)
    if n_classes == 2:
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
    file_config = load_config(args.config)
    config = merge_config(file_config, args)

    # Create output directory
    output_dir = os.path.join("../results", "classifier")
    os.makedirs(output_dir, exist_ok=True)

    # Get files and process
    files = get_filenames(config)
    assert len(files) == 1, "Currently only works with one dataset"

    print("Processing files:", files)
    
    # Process dataset
    X, y, class_names = construct_instrument_dataset(
        files[0],
        by_filename=args.by_filename,
        pattern=args.instrument_pattern,
        verbose=args.verbose
    )
    
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