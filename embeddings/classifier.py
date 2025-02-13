from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os
import yaml
import argparse
from datetime import datetime
import hashlib
from h5_processor import H5DataProcessor, DatasetConfig
from models import RatingsClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(description="Classification of embeddings")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to process')
    parser.add_argument('--output', help='Output filename for the results')
    parser.add_argument('--verbose', '-v', action="store_true", help="Extra print statements for debugging")
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def merge_config(file_config, args):
    if args.datasets:
        file_config['datasets'] = [d for d in file_config['datasets'] if d['dataset'] in args.datasets]
    return file_config

def compute_metrics(y_test, y_pred, y_prob, is_binary=False):
    """Compute classification metrics."""
    metrics = {
        "f1": f1_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "per_class": {
            "precision": precision_score(y_test, y_pred, average=None).tolist(),
            "recall": recall_score(y_test, y_pred, average=None).tolist(),
            "f1": f1_score(y_test, y_pred, average=None).tolist()
        }
    }

    if is_binary:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
        except Exception:
            metrics["roc_auc"] = None
    
    metrics["confidence"] = {
        "mean": float(np.mean(np.max(y_prob, axis=1))),
        "std": float(np.std(np.max(y_prob, axis=1)))
    }
    
    return metrics

def get_model_config(model):
    """Extract model configuration and hyperparameters.
    Gets all instance attributes that don't start with underscore.
    Handles numpy arrays and other non-serializable types.
    """
    def make_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (int, float, str, bool, type(None))):
            return val
        else:
            return str(val)

    # Get all instance attributes that don't start with underscore
    attributes = {
        key: make_serializable(value) 
        for key, value in vars(model).items() 
        if not key.startswith('_')
    }

    return {
        "type": model.__class__.__name__,
        "hyperparameters": attributes
    }

def train_evaluate_model(X, y, model, verbose=False):
    """Train and evaluate the classification model."""
    if verbose:
        print("\nInitial class distribution:")
        for i in range(len(np.unique(y))):
            print(f"Class {i}: {np.sum(y == i)} samples")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train, verbose)

    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    if verbose:
        print("\nPrediction distribution:")
        for i in range(len(np.unique(y))):
            print(f"Class {i}: {np.sum(y_pred == i)} predictions")
    
    # Compute metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred, y_prob, is_binary=len(np.unique(y)) == 2)
    
    return conf_matrix, metrics

def save_results(config, matrix, metrics, model_config, class_names, output_dir):
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "model_configuration": model_config,
        "confusion_matrix": matrix.tolist(),
        "metrics": metrics,
        "class_names": class_names
    }
    
    # Generate hash from all results
    results_str = json.dumps(results, sort_keys=True)
    results_hash = hashlib.md5(results_str.encode()).hexdigest()[:8]
    
    # Create filename with timestamp and hash
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"classifier_{timestamp}_{results_hash}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filepath

def main():
    # Parse arguments and load config
    args = parse_arguments()
    file_config = load_config(args.config)
    config = merge_config(file_config, args)

    # Create output directory
    output_dir = os.path.join("../results", "classifier")
    os.makedirs(output_dir, exist_ok=True)

    # Process datasets using H5DataProcessor
    processor = H5DataProcessor(verbose=args.verbose)
    datasets = processor.process_configs(config['datasets'])
    X, y, class_names = processor.combine_datasets(datasets)
    
    # Create and configure model
    model = RatingsClassifier(max_iter=100)
    model_config = get_model_config(model)
    
    # Train and evaluate model
    conf_matrix, metrics = train_evaluate_model(X, y, model, verbose=args.verbose)

    # Save results
    output_path = save_results(config, conf_matrix, metrics, model_config, class_names, output_dir)
    print(f"\nResults have been saved to '{output_path}'")

if __name__ == "__main__":
    main()