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
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Ratio of data to use for testing')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', default='../results/classifier', help='Output directory')
    parser.add_argument('--verbose', '-v', action="store_true", help="Extra print statements for debugging")
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
    
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

def train_evaluate_model(train_data, test_data, model, verbose=False):
    """Train and evaluate the classification model using pre-split data."""
    # Create label mapping for string class labels
    unique_labels = sorted(set(train_data.labels + test_data.labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    X_train = train_data.embeddings
    y_train = np.array([label_to_idx[label] for label in train_data.labels])
    X_test = test_data.embeddings
    y_test = np.array([label_to_idx[label] for label in test_data.labels])
    
    if verbose:
        print("\nTraining set class distribution:")
        for i, label in enumerate(unique_labels):
            print(f"Class {i} ({label}): {np.sum(y_train == i)} samples")
        print("\nTest set class distribution:")
        for i, label in enumerate(unique_labels):
            print(f"Class {i} ({label}): {np.sum(y_test == i)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)

    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    if verbose:
        print("\nPrediction distribution:")
        for i, label in enumerate(unique_labels):
            print(f"Class {i} ({label}): {np.sum(y_pred == i)} predictions")
    
    # Compute metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred, y_prob, is_binary=len(unique_labels) == 2)
    
    # Add split information to metrics
    metrics['split_info'] = {
        'train_size': len(y_train),
        'test_size': len(y_test),
        'train_class_distribution': [int(np.sum(y_train == i)) for i in range(len(unique_labels))],
        'test_class_distribution': [int(np.sum(y_test == i)) for i in range(len(unique_labels))],
        'label_mapping': {str(label): int(idx) for label, idx in label_to_idx.items()}
    }
    
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
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process datasets using H5DataProcessor
    processor = H5DataProcessor(verbose=args.verbose)
    all_train_data = []
    all_val_data = []
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
        train_data, nontrain_data = processor.get_train_test_split(
            dataset, 
            test_ratio=(args.test_ratio+args.val_ratio),
            random_seed=args.random_seed
        )

        # Split the dataset, again!
        val_data, test_data = processor.get_train_test_split(
            nontrain_data,
            test_ratio=args.test_ratio / (args.val_ratio + args.test_ratio)
        )
        
        all_train_data.append(train_data)
        all_val_data.append(val_data)
        all_test_data.append(test_data)
        class_names.update(dataset.labels)
    
    # Combine datasets
    combined_train = ProcessedDataset(
        embeddings=np.vstack([d.embeddings for d in all_train_data]),
        labels=[l for d in all_train_data for l in d.labels],
        filenames=[f for d in all_train_data for f in d.filenames],
        name="combined train",
        num_samples=sum(d.num_samples for d in all_train_data)
    )

    combined_val = ProcessedDataset(
        embeddings=np.vstack([d.embeddings for d in all_val_data]),
        labels=[l for d in all_val_data for l in d.labels],
        filenames=[f for d in all_val_data for f in d.filenames],
        name="combined val",
        num_samples=sum(d.num_samples for d in all_val_data)
    )
    
    combined_test = ProcessedDataset(
        embeddings=np.vstack([d.embeddings for d in all_test_data]),
        labels=[l for d in all_test_data for l in d.labels],
        filenames=[f for d in all_test_data for f in d.filenames],
        name="combined test",
        num_samples=sum(d.num_samples for d in all_test_data)
    )
    
    # Create and configure model
    model = LogisticRegression(penalty=None, C=1, solver='saga', random_state=42, verbose=args.verbose)
    model_config = get_model_config(model)

    # Train and evaluate model
    conf_matrix, metrics = train_evaluate_model(combined_train, combined_test, model, args.verbose)

    # Save results
    output_path = save_results(config, conf_matrix, metrics, model_config, sorted(class_names), output_dir)
    print(f"\nResults have been saved to '{output_path}'")

if __name__ == "__main__":
    main()