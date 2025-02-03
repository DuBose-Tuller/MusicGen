from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import os
import yaml
import argparse
from datetime import datetime
import hashlib
from h5_processor import H5DataProcessor, DatasetConfig, ProcessedDataset
from pathlib import Path

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

def train_evaluate_model(train_data, test_data, verbose=False):
    """Train and evaluate the classification model using pre-split data."""
    X_train = train_data.embeddings
    y_train = np.array([int(label) for label in train_data.labels])
    X_test = test_data.embeddings
    y_test = np.array([int(label) for label in test_data.labels])
    
    if verbose:
        print("\nTraining set class distribution:")
        for i in range(len(np.unique(y_train))):
            print(f"Class {i}: {np.sum(y_train == i)} samples")
        print("\nTest set class distribution:")
        for i in range(len(np.unique(y_test))):
            print(f"Class {i}: {np.sum(y_test == i)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        tol=1e-6,
        C=0.1,
        solver='lbfgs'
    ).fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    if verbose:
        print("\nPrediction distribution:")
        for i in range(len(np.unique(y_test))):
            print(f"Class {i}: {np.sum(y_pred == i)} predictions")
    
    # Compute metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred, y_prob, is_binary=len(np.unique(y_test)) == 2)
    
    # Add split information to metrics
    metrics['split_info'] = {
        'train_size': len(y_train),
        'test_size': len(y_test),
        'train_class_distribution': [int(np.sum(y_train == i)) for i in range(len(np.unique(y_train)))],
        'test_class_distribution': [int(np.sum(y_test == i)) for i in range(len(np.unique(y_test)))]
    }
    
    return conf_matrix, metrics

def generate_output_filename(config):
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"classifier_{timestamp}_{config_hash}.json"

def save_results(config, matrix, metrics, class_names, output_path):
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "confusion_matrix": matrix.tolist(),
        "metrics": metrics,
        "class_names": class_names
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Parse arguments and load config
    args = parse_arguments()
    file_config = load_config(args.config)
    config = merge_config(file_config, args)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process datasets using H5DataProcessor
    processor = H5DataProcessor(verbose=args.verbose)
    all_train_data = []
    all_test_data = []
    class_names = set()

    # Process each dataset and split
    for dataset_config in config['datasets']:
        if args.verbose:
            print(f"\nProcessing dataset: {dataset_config['dataset']}")
        
        dataset = processor.process_h5_file(
            processor.get_embedding_path(DatasetConfig(**dataset_config)),
            DatasetConfig(**dataset_config)
        )
        
        # Split the dataset
        train_data, test_data = processor.get_train_test_split(
            dataset, 
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
        
        all_train_data.append(train_data)
        all_test_data.append(test_data)
        class_names.update(dataset.labels)
    
    # Combine datasets
    combined_train = ProcessedDataset(
        embeddings=np.vstack([d.embeddings for d in all_train_data]),
        labels=[l for d in all_train_data for l in d.labels],
        filenames=[f for d in all_train_data for f in d.filenames],
        name="combined",
        num_samples=sum(d.num_samples for d in all_train_data)
    )
    
    combined_test = ProcessedDataset(
        embeddings=np.vstack([d.embeddings for d in all_test_data]),
        labels=[l for d in all_test_data for l in d.labels],
        filenames=[f for d in all_test_data for f in d.filenames],
        name="combined",
        num_samples=sum(d.num_samples for d in all_test_data)
    )

    # Train and evaluate model
    conf_matrix, metrics = train_evaluate_model(combined_train, combined_test, args.verbose)

    # Save results
    output_path = output_dir / generate_output_filename(config)
    save_results(config, conf_matrix, metrics, sorted(class_names), output_path)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()